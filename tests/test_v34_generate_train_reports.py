from __future__ import annotations

import copy
import inspect
import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pytest

import research_v34_generate_train_reports as reports


def _fixed_candidates() -> list[dict]:
    return reports.build_fixed_candidates(
        reports.load_and_validate_plan(),
        reports.load_control_source(),
    )


def _copy_authoritative(root: Path) -> Path:
    root.mkdir()
    for name in reports.AUTHORITATIVE_FILENAMES:
        shutil.copyfile(reports.EXPECTED_RESULT_DIR / name, root / name)
    return root


def _repin_authoritative(
    monkeypatch: pytest.MonkeyPatch,
    result_dir: Path,
) -> None:
    monkeypatch.setattr(
        reports,
        "EXPECTED_AUTHORITATIVE_SHA256",
        {
            name: reports._sha256(result_dir / name)
            for name in reports.AUTHORITATIVE_FILENAMES
        },
    )


def test_fixed_authoritative_identity_and_default_output_are_pinned():
    assert reports.EXPECTED_RESULT_DIR == Path(
        "results/strategy_opt_20260721/"
        "v34_same_calendar_month_return_fixed_ablation"
    )
    assert reports.DEFAULT_OUTPUT_DIR == Path(
        "results/strategy_opt_20260721/"
        "v34_same_calendar_month_return_fixed_ablation_train_reports"
    )
    expected = {
        "summary.json": (
            "4ceb3e338877a524b17d11b39993d20f480322adf85dda7e3d204d8c5ad6599a"
        ),
        "all_results.json": (
            "0018bb33056855ee472a2b43c1f3c6a20b43cc9dcde15376265d55922aea3de1"
        ),
        "run_metadata.json": (
            "8b2235c91acf24255fce703bcec5fa148f8c622520131c13a18257c0b9715c70"
        ),
        "daily_returns.npz": (
            "8daaf1bf9c11f30ec0a6c60e66ebf30459bcf9234f534bf6c699142454481a8c"
        ),
    }
    assert reports.EXPECTED_AUTHORITATIVE_SHA256 == expected
    assert {
        name: reports._sha256(reports.EXPECTED_RESULT_DIR / name)
        for name in reports.AUTHORITATIVE_FILENAMES
    } == expected


def test_checked_in_authoritative_result_loads_as_two_rejected_fixed_arms():
    loaded = reports.load_authoritative_v34(
        reports.EXPECTED_RESULT_DIR,
        _fixed_candidates(),
    )
    assert list(loaded["rows"]) == ["control", "candidate"]
    assert loaded["summary"]["status"] == reports.REJECTED_STATUS
    assert loaded["summary"]["decision"]["passes_all_primary_gates"] is False
    assert loaded["rows"]["candidate"]["passes_all_primary_gates"] is False
    assert loaded["series"]["dates"][0] == reports.TRAIN_FIRST_DATE
    assert loaded["series"]["dates"][-1] == reports.TRAIN_LAST_DATE
    assert set(loaded["hashes"]) == set(reports.AUTHORITATIVE_FILENAMES)


def test_authoritative_loader_rejects_any_pinned_artifact_byte_drift(
    tmp_path: Path,
):
    result_dir = _copy_authoritative(tmp_path / "authoritative")
    summary_path = result_dir / "summary.json"
    summary_path.write_bytes(summary_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="artifact SHA mismatch: summary.json"):
        reports.load_authoritative_v34(result_dir, _fixed_candidates())


def test_rejected_selection_cannot_be_rewritten_even_if_fixture_is_rehashed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    result_dir = _copy_authoritative(tmp_path / "authoritative")
    summary_path = result_dir / "summary.json"
    all_results_path = result_dir / "all_results.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    all_results = json.loads(all_results_path.read_text(encoding="utf-8"))
    summary["fixed_candidate"]["passes_all_primary_gates"] = True
    all_results["rows"][1]["passes_all_primary_gates"] = True
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    all_results_path.write_text(json.dumps(all_results), encoding="utf-8")
    _repin_authoritative(monkeypatch, result_dir)

    with pytest.raises(ValueError, match="rejected candidate was altered"):
        reports.load_authoritative_v34(result_dir, _fixed_candidates())


def test_authoritative_config_must_equal_both_fixed_arms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    result_dir = _copy_authoritative(tmp_path / "authoritative")
    summary_path = result_dir / "summary.json"
    all_results_path = result_dir / "all_results.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    all_results = json.loads(all_results_path.read_text(encoding="utf-8"))
    summary["fixed_candidate"]["individual_config"]["buy_n"] = 31
    all_results["rows"][1]["individual_config"]["buy_n"] = 31
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    all_results_path.write_text(json.dumps(all_results), encoding="utf-8")
    _repin_authoritative(monkeypatch, result_dir)

    with pytest.raises(ValueError, match="candidate individual_config differs"):
        reports.load_authoritative_v34(result_dir, _fixed_candidates())


def test_run_refuses_existing_output_before_reading_any_input(tmp_path: Path):
    output = tmp_path / "already_exists"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        reports.run(
            output,
            expected_result_dir=tmp_path / "missing_authoritative",
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
    result, discarded = reports._load_physically_truncated_training_panel(800)
    assert result is truncated
    assert discarded == 7
    assert calls[0][0] == "load"
    assert calls[1] == ("truncate",)


def test_training_panel_rejects_a_surviving_holdout_row(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(reports, "load_runtime_npz", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        reports,
        "physically_truncate_runtime",
        lambda _loaded: (
            {
                "trade_dates": np.asarray(
                    ["2018-12-28", "2019-01-02"], dtype="datetime64[D]"
                )
            },
            0,
        ),
    )
    with pytest.raises(ValueError, match="wrong last date"):
        reports._load_physically_truncated_training_panel(800)


def test_factor_work_occurs_only_after_the_truncated_loader_is_gone():
    source = inspect.getsource(reports.run)
    load_at = source.index("_load_physically_truncated_training_panel(max_history)")
    factor_at = source.index("compute_research_arrays(")
    assert load_at < factor_at
    helper_source = inspect.getsource(
        reports._load_physically_truncated_training_panel
    )
    truncate_at = helper_source.index("physically_truncate_runtime(loaded)")
    delete_at = helper_source.index("del loaded")
    return_at = helper_source.index("return data")
    assert truncate_at < delete_at < return_at


def test_exact_series_comparison_rejects_one_bit_and_one_count_of_drift():
    expected = np.asarray([0.0, 1.0, -2.0], dtype=np.float64)
    reports._require_exact_series(
        name="returns", actual=expected.copy(), expected=expected
    )
    changed = expected.copy()
    changed[1] = np.nextafter(changed[1], np.inf)
    with pytest.raises(ValueError, match="row 1"):
        reports._require_exact_series(
            name="returns", actual=changed, expected=expected
        )

    counts = np.asarray([50, 100, 200], dtype=np.int64)
    changed_counts = counts.copy()
    changed_counts[2] += 1
    with pytest.raises(ValueError, match="row 2"):
        reports._require_exact_series(
            name="selectable counts", actual=changed_counts, expected=counts
        )


def _trade_row(*, price: float = 10.0, field: str = "open") -> dict:
    return {
        "code": "000001.SZ",
        "action": "buy",
        "date": "2018-12-28",
        "trade_date": date(2018, 12, 28),
        "price": np.float64(price),
        "price_field": field,
        "volume": np.int64(100),
        "amount": np.float64(price * 100),
    }


def test_per_trade_audit_records_exact_runtime_open_for_every_row():
    audit = reports._audit_trade_log_exact_open(
        role="control",
        label=reports.CONTROL_LABEL,
        trade_log=[_trade_row()],
        open_price=np.asarray([[10.0]], dtype=np.float64),
        date_indices={"2018-12-28": 0},
        stock_indices={"000001.SZ": 0},
    )
    assert audit["passes"] is True
    assert audit["trade_rows"] == 1
    assert not any(audit["issues"].values())
    assert audit["comparison"].endswith("no tolerance")
    assert audit["per_trade_checks"] == [
        {
            "row_index": 0,
            "trade_date": "2018-12-28",
            "code": "000001.SZ",
            "action": "buy",
            "price_field": "open",
            "reported_price": 10.0,
            "runtime_open": 10.0,
            "exact_runtime_open_match": True,
            "passes": True,
        }
    ]


def test_per_trade_audit_rejects_one_bit_of_open_drift_and_wrong_field():
    changed = np.nextafter(np.float64(10.0), np.inf)
    audit = reports._audit_trade_log_exact_open(
        role="candidate",
        label=reports.CANDIDATE_LABEL,
        trade_log=[_trade_row(price=float(changed)), _trade_row(field="close")],
        open_price=np.asarray([[10.0]], dtype=np.float64),
        date_indices={"2018-12-28": 0},
        stock_indices={"000001.SZ": 0},
    )
    assert audit["passes"] is False
    assert audit["issues"]["open_price_mismatches"] == 1
    assert audit["issues"]["non_open_price_field"] == 1
    assert audit["per_trade_checks"][0]["exact_runtime_open_match"] is False
    assert audit["per_trade_checks"][1]["exact_runtime_open_match"] is True


def test_report_payload_is_full_and_declares_train_isolation(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(reports, "compute_per_year_metrics", lambda *_: [])
    monkeypatch.setattr(reports, "compute_hs300_cumulative_returns", lambda *_: [])
    trade_log = [_trade_row()]
    result = {
        "total_return": 1.0,
        "daily_returns": [1.0],
        "cumulative_returns": [1.0],
        "trade_log": trade_log,
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
        stock_pool_size=100,
    )
    assert payload["trade_log"] is trade_log
    assert payload["rebalance_rule"]["price_field"] == "open"
    assert payload["report_metadata"]["stock_pool_size"] == 100
    assert payload["report_metadata"][
        "runtime_physically_truncated_before_factors"
    ] is True
    assert payload["report_metadata"][
        "strategy_holdout_available_to_factor_or_worker"
    ] is False
    assert payload["report_metadata"][
        "authoritative_selection_status"
    ] == reports.REJECTED_STATUS
    assert payload["report_metadata"]["kline_source"] == (
        "physically_truncated_training_runtime"
    )
    assert payload["report_metadata"]["kline_first_date"] == (
        reports.TRAIN_FIRST_DATE
    )
    assert payload["report_metadata"]["kline_last_date"] == (
        reports.TRAIN_LAST_DATE
    )


def test_real_reports_decode_to_training_only_klines_for_both_arms(
    tmp_path: Path,
):
    trade_log = [
        {
            "code": "000001.SZ",
            "action": "buy",
            "trade_date": "2010-01-04",
            "price": 10.0,
            "price_field": "open",
            "volume": 100,
        },
        {
            "code": "000001.SZ",
            "action": "sell",
            "trade_date": "2018-12-28",
            "price": 12.0,
            "price_field": "open",
            "volume": 100,
        },
    ]
    runtime_dates = np.asarray(
        ["2009-12-31", "2010-01-04", "2014-06-30", "2018-12-28"],
        dtype="datetime64[D]",
    )
    data = {
        "trade_dates": runtime_dates,
        "open": np.asarray([[9.0], [10.0], [11.0], [12.0]]),
        "high": np.asarray([[9.1], [10.1], [11.1], [12.1]]),
        "low": np.asarray([[8.9], [9.9], [10.9], [11.9]]),
        "close": np.asarray([[9.0], [10.0], [11.0], [12.0]]),
        "amount": np.asarray([[900.0], [1000.0], [1100.0], [1200.0]]),
    }
    report_payload = {
        "individual_config": {"weights": {"factor": 1.0}},
        "total_return": 20.0,
        "daily_returns": [0.0, 20.0],
        "cumulative_returns": [0.0, 20.0],
        "trade_dates": ["2010-01-04", "2018-12-28"],
        "trade_log": trade_log,
        "daily_snapshots": [],
        "positions": [],
        "cleared_positions": [],
        "delist_events": [],
        "stock_name_map": {"000001.SZ": "样本"},
        "holding_stats": {},
        "metrics": {},
        "hs300_returns": [],
        "init_cash": 1_000_000.0,
        "final_asset": 1_200_000.0,
        "period": {
            "start": reports.TRAIN_FIRST_DATE,
            "end": reports.TRAIN_LAST_DATE,
            "trade_start": reports.TRAIN_FIRST_DATE,
            "trade_end": reports.TRAIN_LAST_DATE,
        },
    }
    original_collector = reports.single_report_module._collect_kline_payload
    for label in (reports.CONTROL_LABEL, reports.CANDIDATE_LABEL):
        report_path = reports._generate_train_capped_single_report(
            copy.deepcopy(report_payload),
            tmp_path / label,
            data,
            {"000001.SZ": 0},
        )
        assert reports.single_report_module._collect_kline_payload is original_collector
        decoded = reports._decode_html_kline_payload(report_path)
        assert decoded["000001.SZ"]["d"] == [
            "2010-01-04",
            "2014-06-30",
            "2018-12-28",
        ]
        assert decoded["000001.SZ"]["episodes"][0]["window_start"] == (
            reports.TRAIN_FIRST_DATE
        )
        assert decoded["000001.SZ"]["episodes"][0]["window_end"] == (
            reports.TRAIN_LAST_DATE
        )
        audit = reports._audit_html_kline_training_bounds(
            report_path, {"000001.SZ"}
        )
        assert audit["passes"] is True
        assert audit["actual_first_date"] == reports.TRAIN_FIRST_DATE
        assert audit["actual_last_date"] == reports.TRAIN_LAST_DATE


def test_record_and_trade_json_serialize_execution_dates(tmp_path: Path):
    record_path = tmp_path / "record.json"
    reports._write_record(
        record_path,
        {"weights": {"factor": 1.0}, "buy_n": 30, "sell_m": 30},
        {
            "daily_returns": np.asarray([0.5]),
            "daily_snapshots": [{"raw_buy_n_list": ["000001.SZ"]}],
            "round_trip_count": 0,
        },
        ["2018-12-28"],
        {},
    )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["sell_m"] == 30
    assert record["factor_run_registry_side_effect"] is False

    trades_path = tmp_path / "trades.json"
    reports._write_json(trades_path, {"trades": [_trade_row()]})
    trades = json.loads(trades_path.read_text(encoding="utf-8"))["trades"]
    assert trades[0]["trade_date"] == "2018-12-28"
    assert trades[0]["price"] == 10.0
    assert trades[0]["volume"] == 100


def test_manifest_hashes_every_generated_file_except_itself(tmp_path: Path):
    output = tmp_path / "output"
    nested = output / reports.CONTROL_LABEL
    nested.mkdir(parents=True)
    (output / "summary.json").write_text("summary", encoding="utf-8")
    (nested / "report.json").write_text("report", encoding="utf-8")
    (nested / "trades.json").write_text("trades", encoding="utf-8")
    (output / "manifest.json").write_text("excluded", encoding="utf-8")
    hashes = reports._generated_file_hashes(output)
    assert set(hashes) == {
        "summary.json",
        f"{reports.CONTROL_LABEL}/report.json",
        f"{reports.CONTROL_LABEL}/trades.json",
    }
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
        groups[name] = {
            "path": str(path),
            "sha256": reports._sha256(path),
        }
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


def test_full_run_source_has_no_holdout_or_lightweight_execution_path():
    source = inspect.getsource(reports.run)
    assert "lightweight=False" in source
    assert "validation" not in source
    assert "test_period" not in source
    assert "holdout_diagnostics" not in source
    assert "_backtest_direct(" in source
    assert "report.json" in source
    assert "record.json" in source
    assert "trades.json" in source
    assert "single.log" in source
    assert "trade_open_audit.json" in source
