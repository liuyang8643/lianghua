from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import research_v34_same_calendar_month_return_ablation as v34
from research_train_robustness import anchored_metrics


def _fixed_date_strings() -> list[str]:
    all_dates = np.arange(
        np.datetime64(v34.TRAIN_FIRST_DATE, "D"),
        np.datetime64(v34.TRAIN_LAST_DATE, "D") + np.timedelta64(1, "D"),
    )
    positions = np.linspace(
        0,
        len(all_dates) - 1,
        v34.EXPECTED_TRAIN_DAYS,
        dtype=np.int64,
    )
    selected = all_dates[positions]
    assert len(np.unique(selected)) == v34.EXPECTED_TRAIN_DAYS
    return selected.astype(str).tolist()


def _control_canary_row() -> dict:
    return {
        "role": "control",
        "label": v34.CONTROL_LABEL,
        "full_calmar": v34.CONTROL_CANARY["full_calmar"],
        "fold_calmars": copy.deepcopy(v34.CONTROL_CANARY["fold_calmars"]),
        "worst_fold_calmar": v34.CONTROL_CANARY["worst_fold_calmar"],
        "robust_score": v34.CONTROL_CANARY["robust_score"],
        "average_exposure": v34.CONTROL_CANARY["average_exposure"],
    }


def _gate_row(*, role: str, label: str) -> dict:
    return {
        "candidate_index": 0 if role == "control" else 1,
        "role": role,
        "label": label,
        "full_calmar": 3.0,
        "fold_calmars": {
            "2010-2012": 1.6,
            "2013-2015": 4.0,
            "2016-2018": 1.7,
        },
        "worst_fold_calmar": 1.6,
        "robust_score": 2.3,
        "average_exposure": 0.50,
        "annualized": 30.0,
        "max_drawdown": -10.0,
        "sharpe": 1.8,
        "terminal_nav": 10.0,
        "total_return": 900.0,
        "minimum_daily_selectable_candidates": 30,
    }


def test_plan_and_fixed_candidate_config_are_exact():
    plan = v34.load_and_validate_plan()
    source = v34.load_control_source()
    candidates = v34.build_fixed_candidates(plan, source)

    assert [(row["role"], row["label"]) for row in candidates] == [
        ("control", v34.CONTROL_LABEL),
        ("candidate", v34.CANDIDATE_LABEL),
    ]
    assert [row["candidate_index"] for row in candidates] == [0, 1]
    assert candidates[0]["individual_config"] == source["individual_config"]

    control = candidates[0]["individual_config"]
    candidate = candidates[1]["individual_config"]
    assert control["weights"] == v34.CONTROL_WEIGHTS
    assert candidate["weights"] == v34.CANDIDATE_WEIGHTS
    assert tuple(candidate["weights"]) == v34.ALL_FACTOR_NAMES
    control_without_weights = copy.deepcopy(control)
    candidate_without_weights = copy.deepcopy(candidate)
    control_without_weights.pop("weights")
    candidate_without_weights.pop("weights")
    assert candidate_without_weights == control_without_weights
    assert candidate["trend_risk_overlay"]["mode"] == "dual_completed"
    assert candidate["buy_n"] == candidate["sell_m"] == 30
    assert tuple(candidate["stock_pool"]) == ("60", "00", "30")
    assert v34._new_factor_class().hist_days == 800
    assert plan["ga_authorized"] is False


def test_plan_rejects_changed_candidate_or_structural_gate(tmp_path: Path):
    payload = json.loads(v34.PLAN_PATH.read_text(encoding="utf-8"))
    payload["fixed_candidate"]["weights"][v34.NEW_FACTOR_NAME] = 0.2
    changed = tmp_path / "changed_candidate.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fixed candidate semantics changed"):
        v34.load_and_validate_plan(changed)

    payload = json.loads(v34.PLAN_PATH.read_text(encoding="utf-8"))
    payload["primary_gates"]["minimum_daily_selectable_candidates"] = 29
    changed = tmp_path / "changed_gate.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="primary gates changed"):
        v34.load_and_validate_plan(changed)


def test_pool_local_ranking_preserves_datetime64_and_ignores_outside_pool():
    class DateAwareFactor:
        hist_days = 1

        def calc_batch(self, panel):
            assert np.issubdtype(panel["trade_dates"].dtype, np.datetime64)
            return np.asarray([[1.0, 2.0, 100.0], [3.0, 1.0, 200.0]])

    class AllFilter:
        hist_days = 1

        def calc_batch(self, panel):
            return np.ones((2, 3), dtype=np.float32)

    data = {
        "stock_codes": np.asarray(["600001", "000001", "688001"]),
        "trade_dates": np.asarray(
            ["2018-01-02", "2018-01-03"], dtype="datetime64[D]"
        ),
    }
    arrays, score_keys = v34.compute_research_arrays(
        data,
        [DateAwareFactor],
        [AllFilter],
    )
    assert score_keys == {"DateAwareFactor"}
    np.testing.assert_allclose(
        arrays["DateAwareFactor"],
        np.asarray([[0.5, 1.0, 0.0], [1.0, 0.5, 0.0]], dtype=np.float32),
    )


def test_runtime_is_physically_copied_and_deleted_before_factor_computation():
    dates = np.asarray(
        ["2018-12-27", "2018-12-28", "2019-01-02", "2019-01-03"],
        dtype="datetime64[D]",
    )
    open_prices = np.arange(8, dtype=np.float64).reshape(4, 2)
    source = {
        "trade_dates": dates,
        "open": open_prices,
        "close": open_prices + 1.0,
        "stock_codes": np.asarray(["600001", "000001"]),
        "issue_price": np.asarray([1.0, 2.0]),
    }
    truncated, post_train_rows = v34.physically_truncate_runtime(source)
    assert post_train_rows == 2
    np.testing.assert_array_equal(truncated["trade_dates"], dates[:2])
    assert str(truncated["trade_dates"][-1]) == v34.TRAIN_LAST_DATE
    assert not np.shares_memory(truncated["trade_dates"], dates)
    assert not np.shares_memory(truncated["open"], open_prices)

    run_source = inspect.getsource(v34.run)
    truncate_at = run_source.index("physically_truncate_runtime(loaded_data)")
    delete_at = run_source.index("del loaded_data")
    factor_at = run_source.index("compute_research_arrays(")
    assert truncate_at < delete_at < factor_at


def test_candidate_validity_uses_only_nonzero_weight_factors():
    plan = v34.load_and_validate_plan()
    candidates = v34.build_fixed_candidates(plan, v34.load_control_source())
    shape = (2, 3)
    arrays = {
        f"_factor_valid_{name}": np.ones(shape, dtype=bool)
        for name in v34.BASE_FACTOR_NAMES
    }
    new_valid = np.asarray(
        [[True, False, True], [False, True, True]], dtype=bool
    )
    arrays[f"_factor_valid_{v34.NEW_FACTOR_NAME}"] = new_valid
    for name in v34.EXPECTED_FILTER_NAMES:
        arrays[name] = np.ones(shape, dtype=bool)

    control_masks = v34.candidate_filter_masks(
        arrays,
        v34.ALL_FACTOR_NAMES,
        candidates[0]["individual_config"],
    )
    np.testing.assert_array_equal(
        control_masks["_active_factor_intersection"],
        np.ones(shape, dtype=bool),
    )
    candidate_masks = v34.candidate_filter_masks(
        arrays,
        v34.ALL_FACTOR_NAMES,
        candidates[1]["individual_config"],
    )
    np.testing.assert_array_equal(
        candidate_masks["_active_factor_intersection"], new_valid
    )
    v34._verify_candidate_validity_semantics(
        arrays,
        set(v34.ALL_FACTOR_NAMES),
        candidates,
    )


def test_daily_selectable_count_is_pool_active_intersection_plus_filters():
    arrays = {
        "_factor_valid_active": np.asarray(
            [[True, True, True, True], [True, False, True, True]], dtype=bool
        ),
        "_factor_valid_zero": np.zeros((2, 4), dtype=bool),
        "FilterST": np.asarray(
            [[True, True, False, True], [True, True, True, True]], dtype=bool
        ),
        "FilterLowPrice": np.asarray(
            [[True, False, True, True], [True, True, True, True]], dtype=bool
        ),
    }
    config = {
        "weights": {"active": 1.0, "zero": 0.0},
        "filter_factors": {"FilterST": True, "FilterLowPrice": True},
    }
    counts = v34.daily_selectable_counts(
        arrays,
        {"active", "zero"},
        config,
        [0, 1],
        [0, 1, 2],
    )
    np.testing.assert_array_equal(counts, [1, 2])
    # Column 3 passes every mask but is outside the strategy pool and is not counted.
    assert counts[0] != 2


def test_natural_folds_each_start_from_nav_one():
    dates = _fixed_date_strings()
    daily = np.full(len(dates), 0.02, dtype=np.float64)
    date_array = np.asarray(dates)
    for start_year in (2010, 2013, 2016):
        first = np.flatnonzero(date_array >= f"{start_year}-01-01")[0]
        daily[first] = -10.0

    result = v34.natural_calendar_metrics(daily, dates)
    for start_year, label in zip((2010, 2013, 2016), v34.FOLD_LABELS):
        mask = (
            (date_array >= f"{start_year}-01-01")
            & (date_array <= f"{start_year + 2}-12-31")
        )
        expected = anchored_metrics(daily[mask])
        assert result["folds"][label] == expected
        assert result["folds"][label]["max_drawdown"] == pytest.approx(
            -10.0, abs=1e-12
        )


def test_control_canary_uses_absolute_1e_12_tolerance():
    row = _control_canary_row()
    v34.validate_control_canary(row)
    assert v34.CANARY_ABSOLUTE_TOLERANCE == 1e-12

    changed = copy.deepcopy(row)
    changed["fold_calmars"]["2016-2018"] += 2e-12
    with pytest.raises(ValueError, match="control canary mismatch"):
        v34.validate_control_canary(changed)


def test_primary_gates_are_strict_and_structural_failure_cannot_freeze():
    control = _gate_row(role="control", label=v34.CONTROL_LABEL)
    control["robust_score"] = 2.0
    control["fold_calmars"]["2010-2012"] = 1.4
    control["fold_calmars"]["2016-2018"] = 1.3
    candidate = _gate_row(role="candidate", label=v34.CANDIDATE_LABEL)
    checks = v34.primary_gate_checks(candidate, control)
    assert checks and all(checks.values())

    equal = copy.deepcopy(candidate)
    equal["robust_score"] = control["robust_score"]
    assert v34.primary_gate_checks(equal, control)[
        "robust_score_strictly_exceeds_control"
    ] is False

    too_narrow = copy.deepcopy(candidate)
    too_narrow["minimum_daily_selectable_candidates"] = 29
    failed = v34.primary_gate_checks(too_narrow, control)
    assert failed["minimum_daily_selectable_candidates_at_least_30"] is False
    annotated, decision = v34.annotate_primary_decision([control, too_narrow])
    assert annotated[1]["passes_all_primary_gates"] is False
    assert decision["status"] == "fixed_candidate_rejected_at_primary_gates"
    assert decision["configuration_freeze_allowed"] is False
    assert decision["configuration_frozen"] is False
    assert decision["holdout_open_allowed"] is False


def test_metadata_hashes_and_outputs_are_complete_and_refuse_overwrite(
    tmp_path: Path,
):
    paths = {}
    for name in ("plan", "runtime", "script", "candidates", "factor"):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode("ascii"))
        paths[name] = path

    metadata = v34.build_run_metadata(
        plan_path=paths["plan"],
        runtime_path=paths["runtime"],
        script_path=paths["script"],
        candidate_source_path=paths["candidates"],
        factor_source_paths={v34.NEW_FACTOR_NAME: paths["factor"]},
        elapsed_seconds=1.25,
        loader_post_train_rows=7,
        runtime_panel_first_date="2006-10-09",
        runtime_panel_last_date=v34.TRAIN_LAST_DATE,
        rank_pool_column_count=2,
        runtime_column_count=3,
    )
    assert metadata["preregistered_plan_sha256"] == v34._sha256(paths["plan"])
    assert metadata["candidate_source_sha256"] == v34._sha256(
        paths["candidates"]
    )
    assert metadata["runtime_sha256"] == v34._sha256(paths["runtime"])
    assert metadata["research_script_sha256"] == v34._sha256(paths["script"])
    assert metadata["factor_source_sha256"][v34.NEW_FACTOR_NAME]["sha256"] == (
        v34._sha256(paths["factor"])
    )
    assert metadata["strategy_holdout_available_to_factor_or_worker"] is False
    assert metadata["ga_used"] is False
    assert metadata["cost_stress_run"] is False
    assert metadata["configuration_frozen"] is False

    output = tmp_path / "output"
    rows = [{"role": "control"}, {"role": "candidate"}]
    dates = ["2018-12-27", "2018-12-28"]
    daily_returns = {
        "control": np.asarray([0.0, 1.0]),
        "candidate": np.asarray([1.0, 0.0]),
    }
    daily_exposures = {
        "control": np.asarray([0.5, 0.6]),
        "candidate": np.asarray([0.7, 0.8]),
    }
    daily_selectable = {
        "control": np.asarray([100, 101]),
        "candidate": np.asarray([80, 81]),
    }
    v34.write_outputs(
        output,
        rows,
        {"status": "test"},
        metadata,
        dates,
        daily_returns,
        daily_exposures,
        daily_selectable,
    )
    assert {path.name for path in output.iterdir()} == {
        "all_results.json",
        "daily_returns.npz",
        "summary.json",
        "run_metadata.json",
    }
    with np.load(output / "daily_returns.npz") as saved:
        np.testing.assert_array_equal(saved["dates"], dates)
        np.testing.assert_allclose(saved["candidate_daily_returns"], [1.0, 0.0])
        np.testing.assert_allclose(saved["candidate_daily_exposures"], [0.7, 0.8])
        np.testing.assert_array_equal(
            saved["candidate_daily_selectable_candidates"], [80, 81]
        )

    with pytest.raises(FileExistsError):
        v34.write_outputs(
            output,
            rows,
            {"status": "test"},
            metadata,
            dates,
            daily_returns,
            daily_exposures,
            daily_selectable,
        )


def test_run_refuses_existing_output_before_loading_runtime(tmp_path: Path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        v34.run(output)
