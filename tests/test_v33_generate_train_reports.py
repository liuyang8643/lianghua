from __future__ import annotations

import copy
import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

import research_v33_generate_train_reports as reports
from research_v33_completed_52week_high_sleeve import build_fixed_blend


def _fixed_dates() -> list[str]:
    all_dates = np.arange(
        np.datetime64(reports.TRAIN_FIRST_DATE, "D"),
        np.datetime64(reports.TRAIN_LAST_DATE, "D") + np.timedelta64(1, "D"),
    )
    positions = np.linspace(
        0,
        len(all_dates) - 1,
        reports.EXPECTED_TRAIN_DAYS,
        dtype=np.int64,
    )
    selected = all_dates[positions]
    assert len(np.unique(selected)) == reports.EXPECTED_TRAIN_DAYS
    return selected.astype(str).tolist()


def _metric_row(
    role: str,
    label: str,
    config: dict | None,
    daily_returns: np.ndarray,
    exposures: np.ndarray,
    dates: list[str],
) -> dict:
    natural = reports.natural_calendar_metrics(daily_returns, dates)
    return {
        "candidate_index": {"control": 0, "standalone": 1, "blend": 2}[role],
        "role": role,
        "label": label,
        "individual_config": copy.deepcopy(config),
        "full_calmar": float(natural["full"]["calmar"]),
        "fold_calmars": {
            fold: float(natural["folds"][fold]["calmar"])
            for fold in reports.FOLD_LABELS
        },
        "worst_fold_calmar": float(natural["worst_fold_calmar"]),
        "robust_score": float(natural["robust_score"]),
        "average_exposure": float(np.mean(exposures)),
    }


def _write_authoritative_fixture(
    root: Path,
    candidates: list[dict],
) -> dict:
    dates = _fixed_dates()
    axis = np.arange(len(dates), dtype=np.float64)
    control_returns = 0.035 + 0.30 * np.sin(axis / 7.0)
    standalone_returns = 0.025 + 0.25 * np.cos(axis / 11.0)
    control_exposures = 0.55 + 0.05 * np.sin(axis / 23.0)
    standalone_exposures = 0.90 + 0.05 * np.cos(axis / 29.0)
    blend_returns, blend_exposures = build_fixed_blend(
        control_returns,
        standalone_returns,
        control_exposures,
        standalone_exposures,
    )
    rows = [
        _metric_row(
            "control",
            reports.CONTROL_LABEL,
            candidates[0]["individual_config"],
            control_returns,
            control_exposures,
            dates,
        ),
        _metric_row(
            "standalone",
            reports.STANDALONE_LABEL,
            candidates[1]["individual_config"],
            standalone_returns,
            standalone_exposures,
            dates,
        ),
        _metric_row(
            "blend",
            reports.BLEND_LABEL,
            None,
            blend_returns,
            blend_exposures,
            dates,
        ),
    ]
    summary = {
        "experiment": reports.EXPERIMENT,
        "selection_scope": "training_only_2010_2018",
        "control_canary_passed": True,
        "control": rows[0],
        "standalone_diagnostics": rows[1],
        "fixed_blend": rows[2],
        "strategy_holdout_available_to_factor_or_worker": False,
        "strategy_holdout_evaluated": False,
    }
    all_results = {"experiment": reports.EXPERIMENT, "rows": rows}
    metadata = {
        "experiment": reports.EXPERIMENT,
        "sealed_holdout": True,
        "strategy_holdout_available_to_factor_or_worker": False,
        "strategy_holdout_evaluated": False,
    }
    root.mkdir()
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (root / "all_results.json").write_text(
        json.dumps(all_results), encoding="utf-8"
    )
    (root / "run_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    np.savez_compressed(
        root / "daily_returns.npz",
        dates=np.asarray(dates, dtype="U10"),
        control_daily_returns=control_returns,
        standalone_daily_returns=standalone_returns,
        blend_daily_returns=blend_returns,
        control_daily_exposures=control_exposures,
        standalone_daily_exposures=standalone_exposures,
        blend_daily_exposures=blend_exposures,
    )
    return {"rows": rows, "summary": summary}


def test_run_refuses_existing_output_before_reading_inputs(tmp_path: Path):
    output = tmp_path / "already_exists"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        reports.run(
            output,
            expected_result_dir=tmp_path / "missing_result",
            plan_path=tmp_path / "missing_plan",
            candidate_source_path=tmp_path / "missing_candidates",
        )


def test_training_panel_is_physically_truncated_before_return(
    monkeypatch: pytest.MonkeyPatch,
):
    loaded = {"sentinel": object()}
    truncated = {
        "trade_dates": np.asarray(
            ["2018-12-27", "2018-12-28"], dtype="datetime64[D]"
        )
    }
    calls = []

    def fake_load(dates, max_lookback):
        calls.append(("load", max_lookback, len(dates)))
        return loaded

    def fake_truncate(value):
        assert value is loaded
        calls.append(("truncate",))
        return truncated, 7

    monkeypatch.setattr(reports, "load_runtime_npz", fake_load)
    monkeypatch.setattr(reports, "physically_truncate_runtime", fake_truncate)
    result, discarded = reports._load_physically_truncated_training_panel(280)
    assert result is truncated
    assert discarded == 7
    assert calls[0][0] == "load"
    assert calls[1] == ("truncate",)


def test_exact_series_comparison_rejects_one_bit_of_drift():
    expected = np.asarray([0.0, 1.0, -2.0], dtype=np.float64)
    reports._require_exact_series(
        name="test", actual=expected.copy(), expected=expected
    )
    changed = expected.copy()
    changed[1] = np.nextafter(changed[1], np.inf)
    with pytest.raises(ValueError, match="row 1"):
        reports._require_exact_series(
            name="test", actual=changed, expected=expected
        )


def test_authoritative_loader_requires_fixed_configs_and_exact_blend(
    tmp_path: Path,
):
    plan = reports.load_and_validate_plan()
    candidates = reports.build_fixed_candidates(
        plan, reports.load_control_source()
    )
    result_dir = tmp_path / "authoritative"
    _write_authoritative_fixture(result_dir, candidates)
    loaded = reports.load_authoritative_v33(result_dir, candidates)
    assert list(loaded["rows"]) == ["control", "standalone", "blend"]
    assert loaded["series"]["dates"][0] == reports.TRAIN_FIRST_DATE
    assert set(loaded["hashes"]) == set(reports.AUTHORITATIVE_FILENAMES)

    summary_path = result_dir / "summary.json"
    all_path = result_dir / "all_results.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    all_results = json.loads(all_path.read_text(encoding="utf-8"))
    summary["standalone_diagnostics"]["individual_config"]["buy_n"] = 31
    all_results["rows"][1]["individual_config"]["buy_n"] = 31
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    all_path.write_text(json.dumps(all_results), encoding="utf-8")
    with pytest.raises(ValueError, match="standalone config differs"):
        reports.load_authoritative_v33(result_dir, candidates)


def test_authoritative_loader_rejects_inexact_saved_blend(tmp_path: Path):
    candidates = reports.build_fixed_candidates(
        reports.load_and_validate_plan(), reports.load_control_source()
    )
    result_dir = tmp_path / "authoritative"
    _write_authoritative_fixture(result_dir, candidates)
    original = result_dir / "daily_returns.npz"
    with np.load(original, allow_pickle=False) as saved:
        arrays = {name: np.asarray(saved[name]).copy() for name in saved.files}
    arrays["blend_daily_returns"][10] = np.nextafter(
        arrays["blend_daily_returns"][10], np.inf
    )
    np.savez_compressed(original, **arrays)
    with pytest.raises(ValueError, match="blend returns.*row 10"):
        reports.load_authoritative_v33(result_dir, candidates)


def test_synthetic_output_contains_daily_nav_but_no_transaction_record(
    tmp_path: Path,
):
    expected_row = {
        "full_calmar": 3.0,
        "worst_fold_calmar": 1.6,
        "robust_score": 2.2,
        "average_exposure": 0.6,
    }
    payload = reports._synthetic_payload(
        dates=["2018-12-27", "2018-12-28"],
        daily_returns=np.asarray([1.0, -0.5]),
        daily_exposures=np.asarray([0.5, 0.6]),
        expected_row=expected_row,
    )
    run_dir = tmp_path / "synthetic"
    written = reports._write_synthetic_outputs(run_dir, payload)
    saved = json.loads(
        (run_dir / "synthetic_blend.json").read_text(encoding="utf-8")
    )
    assert saved["transactions_generated"] is False
    assert "trades" not in saved
    assert "trade_log" not in saved
    assert saved["cumulative_returns_pct"] == pytest.approx([1.0, 0.495])
    assert not (run_dir / "single_report.html").exists()
    page = (run_dir / "synthetic_blend.html").read_text(encoding="utf-8")
    assert "No transaction record exists" in page
    assert written["transactions_generated"] is False


def test_report_payload_preserves_real_trades_and_declares_train_isolation(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(reports, "compute_per_year_metrics", lambda *_: [])
    monkeypatch.setattr(reports, "compute_hs300_cumulative_returns", lambda *_: [])
    result = {
        "total_return": 1.0,
        "daily_returns": [1.0],
        "cumulative_returns": [1.0],
        "trade_log": [{"date": "2018-12-28", "price": 10.0}],
        "daily_snapshots": [],
        "positions": [],
        "cleared_positions": [],
        "delist_events": [],
        "stock_name_map": {},
        "holding_stats": {},
        "executed_buy_count": 1,
        "executed_sell_count": 0,
        "delist_count": 0,
        "round_trip_count": 0,
        "final_asset": 1_010_000.0,
        "cleared_positions_count": 0,
        "current_positions_count": 1,
    }
    payload = reports._report_payload(
        config={"weights": {"factor": 1.0}},
        result=result,
        dates=["2018-12-28"],
        legacy_metrics={},
        identity_hashes={"runtime": {"sha256": "abc"}},
    )
    assert payload["trade_log"] is result["trade_log"]
    assert payload["rebalance_rule"]["price_field"] == "open"
    assert payload["report_metadata"][
        "runtime_physically_truncated_before_factors"
    ] is True
    assert payload["report_metadata"][
        "strategy_holdout_available_to_factor_or_worker"
    ] is False


def test_raw_trade_json_serializes_execution_dates(tmp_path: Path):
    path = tmp_path / "trades.json"
    reports._write_json(
        path,
        {
            "trades": [
                {
                    "trade_date": date(2018, 12, 28),
                    "price": np.float64(10.5),
                    "volume": np.int64(100),
                }
            ]
        },
    )
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["trades"] == [
        {"trade_date": "2018-12-28", "price": 10.5, "volume": 100}
    ]


def test_manifest_hashes_every_generated_file_except_itself(tmp_path: Path):
    output = tmp_path / "output"
    nested = output / "nested"
    nested.mkdir(parents=True)
    (output / "summary.json").write_text("summary", encoding="utf-8")
    (nested / "trades.json").write_text("trades", encoding="utf-8")
    (output / "manifest.json").write_text("excluded", encoding="utf-8")
    hashes = reports._generated_file_hashes(output)
    assert set(hashes) == {"summary.json", "nested/trades.json"}
    assert hashes["summary.json"] == reports._sha256(output / "summary.json")


def test_input_identity_recheck_detects_concurrent_change(tmp_path: Path):
    groups = {}
    for name in (
        "preregistered_plan",
        "candidate_source",
        "runtime",
        "authoritative_adapter",
    ):
        path = tmp_path / f"{name}.txt"
        path.write_text(name, encoding="utf-8")
        groups[name] = {"path": str(path), "sha256": reports._sha256(path)}
    factor = tmp_path / "factor.py"
    result = tmp_path / "summary.json"
    factor.write_text("factor", encoding="utf-8")
    result.write_text("result", encoding="utf-8")
    groups["factor_sources"] = {
        "Factor": {"path": str(factor), "sha256": reports._sha256(factor)}
    }
    groups["authoritative_result"] = {
        "summary.json": {
            "path": str(result),
            "sha256": reports._sha256(result),
        }
    }
    reports._assert_input_hashes_unchanged(groups)
    result.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="input changed during generation"):
        reports._assert_input_hashes_unchanged(groups)
