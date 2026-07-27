from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import research_v36_failure_diagnostics as diagnostics


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _all_keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            found.add(str(key).lower())
            found.update(_all_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_all_keys(child))
    return found


def _formal_dates() -> list[str]:
    with np.load(diagnostics.RESULT_DIR / "daily_returns.npz", allow_pickle=False) as data:
        return [str(value) for value in data["dates"].tolist()]


def test_pinned_authoritative_hashes_and_paths_are_exact():
    assert diagnostics.RESULT_DIR.name == diagnostics.EXPERIMENT
    assert diagnostics.DEFAULT_OUTPUT_PATH.parent == diagnostics.RESULT_DIR.parent
    assert diagnostics.DEFAULT_OUTPUT_PATH.parent != diagnostics.RESULT_DIR
    assert diagnostics.DEFAULT_OUTPUT_PATH.name == (
        "v36_signed_amount_imbalance_fixed_ablation_failure_diagnostics.json"
    )
    expected = {
        diagnostics.RESULT_DIR / "manifest.json": diagnostics.EXPECTED_MANIFEST_SHA256,
        diagnostics.PLAN_PATH: diagnostics.EXPECTED_PLAN_SHA256,
        diagnostics.SOURCE_LOCK_PATH: diagnostics.EXPECTED_SOURCE_LOCK_SHA256,
        diagnostics.RUNTIME_PATH: diagnostics.EXPECTED_RUNTIME_SHA256,
    }
    for path, digest in expected.items():
        assert len(digest) == 64
        assert digest == digest.lower()
        assert _sha256(path) == digest


def test_strict_json_rejects_duplicate_and_nonfinite_values():
    with pytest.raises(ValueError, match="duplicate JSON key"):
        diagnostics._loads_strict_json('{"a": 1, "a": 2}')
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        diagnostics._loads_strict_json('{"a": NaN}')
    with pytest.raises(ValueError, match="non-finite numeric"):
        diagnostics._loads_strict_json('{"a": 1e999}')


def test_manifest_member_hash_failure_precedes_artifact_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(diagnostics, "REPO_ROOT", tmp_path)
    result_dir = tmp_path / diagnostics.EXPERIMENT
    result_dir.mkdir()
    artifact = result_dir / "broken.json"
    _write_json(artifact, {"finite": 1})
    manifest = {
        "experiment": diagnostics.EXPERIMENT,
        "status": diagnostics.EXPECTED_STATUS,
        "input_hashes_verified_at_start_and_end": True,
        "manifest_written_after_every_other_artifact": True,
        "input_source_runtime_sha256": {},
        "generated_artifact_sha256": {"broken.json": "0" * 64},
    }
    manifest_path = result_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    plan_path = tmp_path / "plan.json"
    _write_json(
        plan_path,
        {
            "experiment": diagnostics.EXPERIMENT,
            "selection_scope": "training_only_2010_2018",
        },
    )
    lock_path = tmp_path / "lock.json"
    _write_json(lock_path, {"experiment": diagnostics.EXPERIMENT})
    artifact_parse_called = False

    def forbidden_parse(path: Path) -> object:
        nonlocal artifact_parse_called
        artifact_parse_called = True
        raise AssertionError(path)

    monkeypatch.setattr(diagnostics, "_strict_artifact_parse", forbidden_parse)
    with pytest.raises(ValueError, match="formal artifact hash mismatch"):
        diagnostics.verify_authoritative_inputs(
            result_dir=result_dir,
            manifest_sha256=_sha256(manifest_path),
            plan_path=plan_path,
            plan_sha256=_sha256(plan_path),
            source_lock_path=lock_path,
            source_lock_sha256=_sha256(lock_path),
        )
    assert artifact_parse_called is False


def test_training_date_seal_is_exact_and_rejects_later_or_reordered_dates():
    dates = _formal_dates()
    diagnostics._assert_training_dates(dates)
    later = dates.copy()
    later[-1] = "2019-01-02"
    with pytest.raises(ValueError, match="boundaries|post-training"):
        diagnostics._assert_training_dates(later)
    reordered = dates.copy()
    reordered[100], reordered[101] = reordered[101], reordered[100]
    with pytest.raises(ValueError, match="strictly increasing"):
        diagnostics._assert_training_dates(reordered)


def test_corporate_safe_next_open_formula_uses_official_reference_price():
    result = diagnostics.corporate_safe_next_open_return(
        np.asarray([10.0, 20.0]),
        np.asarray([11.0, 18.0]),
        np.asarray([6.05, 10.45]),
        np.asarray([5.5, 9.5]),
    )
    np.testing.assert_allclose(result, np.asarray([0.21, -0.01]), rtol=0.0, atol=1e-15)


def test_average_rank_and_spearman_ties_are_deterministic():
    values = np.asarray([5.0, 1.0, 1.0, 3.0])
    np.testing.assert_array_equal(
        diagnostics._average_ranks(values), np.asarray([4.0, 1.5, 1.5, 3.0])
    )
    assert diagnostics._spearman(values, values) == pytest.approx(1.0)


def test_exact_drawdown_episode_locks_peak_trough_and_recovery():
    result = diagnostics.exact_drawdown_episodes(
        ["2010-01-04", "2010-01-05", "2010-01-06", "2010-01-07"],
        [10.0, -20.0, 25.0, 1.0],
    )
    assert result["episode_count"] == 1
    episode = result["maximum_episode"]
    assert episode["peak_date"] == "2010-01-04"
    assert episode["start_date"] == "2010-01-05"
    assert episode["trough_date"] == "2010-01-05"
    assert episode["recovery_date"] == "2010-01-06"
    assert episode["drawdown_pct"] == pytest.approx(-20.0)
    assert episode["peak_to_trough_trading_days"] == 1
    assert episode["underwater_trading_days"] == 2


def test_formal_trade_and_top30_regression_canaries():
    dates = _formal_dates()
    records = {
        role: diagnostics._read_strict_json(
            diagnostics.RESULT_DIR / label / "record.json"
        )
        for role, label in diagnostics.ARM_LABELS.items()
    }
    trades = {
        role: diagnostics._read_strict_json(
            diagnostics.RESULT_DIR / label / "trades.json"
        )
        for role, label in diagnostics.ARM_LABELS.items()
    }
    for role in diagnostics.ARM_LABELS:
        diagnostics._validate_record(records[role], role, dates)
        diagnostics._validate_trades(trades[role], role, dates)
    overlap = diagnostics.trade_overlap_diagnostics(trades, dates)[
        "by_fixed_period"
    ]["full_training"]
    assert overlap["control_events"] == 50469
    assert overlap["candidate_events"] == 49541
    assert overlap["shared_events"] == 33744
    assert overlap["union_events"] == 66266
    assert overlap["jaccard"] == pytest.approx(0.5092204146923007)
    top30 = diagnostics.top30_overlap_rank_churn(records, dates)
    full = top30["by_fixed_period"]["full_training"]
    assert full["shared_slots"] == 53944
    assert full["available_slots"] == 65610
    assert full["shared_fraction"] == pytest.approx(0.8221917390641671)
    assert full["control_replacements"] == 9355
    assert full["candidate_replacements"] == 8861
    assert top30["pooled_common_member_absolute_rank_difference"][
        "mean"
    ] == pytest.approx(5.153696425922438)


def test_source_has_no_forbidden_execution_or_search_call_path():
    source_path = diagnostics.REPO_ROOT / "research_v36_failure_diagnostics.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    forbidden_modules = {
        "core.backtest",
        "testback.run_ga",
        "research_train_robustness",
        "research_train_cost_stress",
    }
    assert imported_modules.isdisjoint(forbidden_modules)
    called_names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            called_names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            called_names.add(node.func.attr)
    assert called_names.isdisjoint(
        {
            "_backtest_direct",
            "run_backtest",
            "run_ga",
            "stress_daily_returns",
            "_rolling_blocks",
        }
    )
    lowered = source.lower()
    for forbidden in (
        "residualization",
        "alternate_sign",
        "parameter_grid",
        "weight_grid",
    ):
        assert forbidden not in lowered


def test_output_writer_is_byte_deterministic_and_rejects_nonfinite(tmp_path: Path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    payload = {"z": [3, 2, 1], "a": {"finite": 1.25}}
    diagnostics._write_output(first, payload)
    diagnostics._write_output(second, payload)
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes().endswith(b"\n")
    with pytest.raises(ValueError, match="non-finite"):
        diagnostics._write_output(tmp_path / "bad.json", {"bad": float("nan")})


def test_generated_output_has_no_forbidden_diagnostic_fields():
    assert diagnostics.DEFAULT_OUTPUT_PATH.is_file()
    payload = diagnostics._read_strict_json(diagnostics.DEFAULT_OUTPUT_PATH)
    assert payload["diagnostic_non_selection"] is True
    assert payload["training_scope"] == {
        "days": 2187,
        "first_date": "2010-01-04",
        "fixed_folds": ["2010-2012", "2013-2015", "2016-2018"],
        "fixed_years": [str(year) for year in range(2010, 2019)],
        "last_date": "2018-12-28",
    }
    forbidden_fragments = {
        "rolling",
        "p10",
        "largest_year",
        "concentration",
        "cost_stress",
        "residual",
        "alternate_sign",
        "parameter_grid",
        "weight_grid",
        "sealed_period",
    }
    for key in _all_keys(payload):
        assert not any(fragment in key for fragment in forbidden_fragments), key
    assert payload["integrity"]["strategy_execution_calls"] == 0
    assert payload["integrity"]["search_calls"] == 0
    assert payload["raw_factor_daily_ic"]["last_signal_date"] < "2018-12-28"
    assert len(payload["provenance"]["formal_artifacts"]) == 24
    assert payload["provenance"]["composite_pool_sha256"] == (
        "b353f7c36799f6da7c1612ec396d232be4c832017c4267c5b88fd3046c736133"
    )
    assert payload["provenance"]["selectable_counts_sha256"] == (
        "f634e0a8dde67391030443f7d32f104f596010caf5cf3797ffc2462468b40d40"
    )


def test_generated_output_locks_independently_recomputed_failure_canaries():
    payload = diagnostics._read_strict_json(diagnostics.DEFAULT_OUTPUT_PATH)
    fixed_labels = {
        "full_training",
        *diagnostics.FOLD_RANGES,
        *diagnostics.YEAR_LABELS,
    }
    assert set(payload["raw_factor_daily_ic"]["by_fixed_period"]) == fixed_labels
    correlations = payload["daily_series_correlations"]["by_fixed_period"]
    assert set(correlations) == fixed_labels
    assert correlations["full_training"]["return_pearson"] == pytest.approx(
        0.9860051838439495
    )
    ic = payload["raw_factor_daily_ic"]["by_fixed_period"]
    assert ic["full_training"]["days"] == 2186
    assert ic["full_training"]["mean_daily_spearman"] == pytest.approx(
        -0.023231427496843822
    )
    assert ic["2010-2012"]["mean_daily_spearman"] == pytest.approx(
        -0.020659606941733313
    )
    assert ic["2013-2015"]["mean_daily_spearman"] == pytest.approx(
        -0.0264343444904022
    )
    assert ic["2016-2018"]["mean_daily_spearman"] == pytest.approx(
        -0.022609970689115903
    )
    coverage = payload["raw_factor_coverage_distribution"]["by_fixed_period"][
        "full_training"
    ]
    assert coverage["ranking_pool"]["finite_fraction"] == pytest.approx(
        0.4472593584887921
    )
    assert coverage["actual_composite_pool"]["stock_days"] == 3_688_499
    paired = payload["candidate_only_vs_control_only_fixed_diagnostics"][
        "top30_paired_daily_differences"
    ]["by_fixed_period"]["full_training"]
    assert paired["candidate_minus_control_daily_mean_raw_factor"][
        "mean"
    ] == pytest.approx(0.3600607244474433)
    assert paired["candidate_minus_control_daily_mean_next_open_return"][
        "mean"
    ] == pytest.approx(-0.00032358742586993043)
    episodes = payload["exact_drawdown_episodes"]["arms"]
    control = episodes["control"]["by_fixed_period"]["full_training"][
        "maximum_episode"
    ]
    candidate = episodes["candidate"]["by_fixed_period"]["full_training"][
        "maximum_episode"
    ]
    assert (control["peak_date"], control["trough_date"], control["recovery_date"]) == (
        "2015-06-12",
        "2015-07-28",
        "2015-10-15",
    )
    assert control["peak_to_trough_trading_days"] == 31
    assert control["underwater_trading_days"] == 81
    assert control["drawdown_pct"] == pytest.approx(-17.724585057390353)
    assert (
        candidate["peak_date"],
        candidate["trough_date"],
        candidate["recovery_date"],
    ) == ("2015-06-15", "2015-09-25", "2015-10-27")
    assert candidate["peak_to_trough_trading_days"] == 71
    assert candidate["underwater_trading_days"] == 88
    assert candidate["drawdown_pct"] == pytest.approx(-23.539997764437004)
