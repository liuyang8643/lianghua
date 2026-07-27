from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import research_v33_completed_52week_high_sleeve as v33
from research_train_robustness import anchored_metrics


def _fixed_date_strings() -> list[str]:
    all_dates = np.arange(
        np.datetime64(v33.TRAIN_FIRST_DATE, "D"),
        np.datetime64(v33.TRAIN_LAST_DATE, "D") + np.timedelta64(1, "D"),
    )
    positions = np.linspace(
        0,
        len(all_dates) - 1,
        v33.EXPECTED_TRAIN_DAYS,
        dtype=np.int64,
    )
    selected = all_dates[positions]
    assert len(np.unique(selected)) == v33.EXPECTED_TRAIN_DAYS
    return selected.astype(str).tolist()


def _control_canary_row() -> dict:
    return {
        "role": "control",
        "label": v33.CONTROL_LABEL,
        "full_calmar": v33.CONTROL_CANARY["full_calmar"],
        "fold_calmars": copy.deepcopy(v33.CONTROL_CANARY["fold_calmars"]),
        "worst_fold_calmar": v33.CONTROL_CANARY["worst_fold_calmar"],
        "robust_score": v33.CONTROL_CANARY["robust_score"],
        "average_exposure": v33.CONTROL_CANARY["average_exposure"],
    }


def _gate_row() -> dict:
    return {
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
    }


def test_plan_and_fixed_candidate_semantics_are_exact():
    plan = v33.load_and_validate_plan()
    control_source = v33.load_control_source()
    candidates = v33.build_fixed_candidates(plan, control_source)

    assert [(row["role"], row["label"]) for row in candidates] == [
        ("control", v33.CONTROL_LABEL),
        ("standalone", v33.STANDALONE_LABEL),
    ]
    control = candidates[0]["individual_config"]
    standalone = candidates[1]["individual_config"]
    assert control["weights"] == v33.CONTROL_WEIGHTS
    assert control["trend_risk_overlay"]["mode"] == "dual_completed"
    assert standalone["weights"] == {
        "Completed52WeekHighProximityStrict": 1.0
    }
    assert standalone["buy_n"] == standalone["sell_m"] == 30
    assert standalone["timing_enabled"] is False
    assert "trend_risk_overlay" not in standalone
    assert tuple(standalone["stock_pool"]) == ("60", "00", "30")


def test_plan_rejects_any_other_blend_weight(tmp_path: Path):
    payload = json.loads(v33.PLAN_PATH.read_text(encoding="utf-8"))
    payload["fixed_combination"]["capital_weights"] = {
        "v31_control": 0.70,
        "standalone_trend": 0.30,
    }
    changed = tmp_path / "changed_plan.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="blend weights changed"):
        v33.load_and_validate_plan(changed)


def test_pool_local_ranking_preserves_datetime64_for_factor():
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
        "trade_dates": np.asarray(["2018-01-02", "2018-01-03"], dtype="datetime64[D]"),
    }
    arrays, score_keys = v33.compute_research_arrays(
        data,
        [DateAwareFactor],
        [AllFilter],
    )
    assert score_keys == {"DateAwareFactor"}
    np.testing.assert_allclose(arrays["DateAwareFactor"][:, 2], 0.0)
    np.testing.assert_allclose(
        arrays["DateAwareFactor"],
        np.asarray([[0.5, 1.0, 0.0], [1.0, 0.5, 0.0]], dtype=np.float32),
    )


def test_candidate_validity_uses_only_nonzero_weight_factors():
    arrays = {
        "_factor_valid_active": np.asarray([[True, False], [True, True]]),
        "_factor_valid_zero": np.asarray([[False, False], [False, False]]),
        "FilterST": np.asarray([[True, True], [False, True]]),
    }
    config = {
        "weights": {"active": 1.0, "zero": 0.0},
        "filter_factors": {"FilterST": True},
    }
    masks = v33.candidate_filter_masks(arrays, {"active", "zero"}, config)
    np.testing.assert_array_equal(
        masks["_active_factor_intersection"],
        arrays["_factor_valid_active"],
    )
    np.testing.assert_array_equal(masks["FilterST"], arrays["FilterST"])
    v33._verify_candidate_validity_semantics(
        arrays,
        {"active", "zero"},
        [{"individual_config": config}],
    )


def test_runtime_is_physically_copied_before_holdout_rows():
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
    truncated, post_train_rows = v33.physically_truncate_runtime(source)
    assert post_train_rows == 2
    np.testing.assert_array_equal(
        truncated["trade_dates"], dates[:2]
    )
    assert str(truncated["trade_dates"][-1]) == v33.TRAIN_LAST_DATE
    assert not np.shares_memory(truncated["trade_dates"], dates)
    assert not np.shares_memory(truncated["open"], open_prices)
    np.testing.assert_array_equal(truncated["stock_codes"], source["stock_codes"])


def test_fixed_blend_has_no_weight_parameter_and_is_exact():
    assert tuple(inspect.signature(v33.build_fixed_blend).parameters) == (
        "control_daily_returns",
        "standalone_daily_returns",
        "control_daily_exposures",
        "standalone_daily_exposures",
    )
    returns, exposures = v33.build_fixed_blend(
        [4.0, -2.0],
        [0.0, 6.0],
        [0.8, 0.4],
        [0.2, 1.0],
    )
    np.testing.assert_allclose(returns, [3.0, 0.0])
    np.testing.assert_allclose(exposures, [0.65, 0.55])


def test_natural_folds_are_each_anchored_at_nav_one():
    dates = _fixed_date_strings()
    daily = np.full(len(dates), 0.02, dtype=np.float64)
    date_array = np.asarray(dates)
    for start_year in (2010, 2013, 2016):
        first = np.flatnonzero(date_array >= f"{start_year}-01-01")[0]
        daily[first] = -10.0

    result = v33.natural_calendar_metrics(daily, dates)
    for start_year, label in zip((2010, 2013, 2016), v33.FOLD_LABELS):
        mask = (
            (date_array >= f"{start_year}-01-01")
            & (date_array <= f"{start_year + 2}-12-31")
        )
        expected = anchored_metrics(daily[mask])
        assert result["folds"][label] == expected
        assert result["folds"][label]["max_drawdown"] == pytest.approx(
            -10.0, abs=1e-12
        )


def test_combination_gates_are_absolute_and_strictly_control_relative():
    control = _gate_row()
    control["robust_score"] = 2.0
    control["fold_calmars"]["2010-2012"] = 1.4
    control["fold_calmars"]["2016-2018"] = 1.3
    blend = _gate_row()
    checks = v33.combination_gate_checks(blend, control)
    assert checks and all(checks.values())

    equal = copy.deepcopy(blend)
    equal["robust_score"] = control["robust_score"]
    equal["fold_calmars"]["2010-2012"] = control["fold_calmars"]["2010-2012"]
    equal["fold_calmars"]["2016-2018"] = control["fold_calmars"]["2016-2018"]
    failed = v33.combination_gate_checks(equal, control)
    assert failed["robust_score_strictly_exceeds_control"] is False
    assert failed["fold_2010_2012_strictly_exceeds_control"] is False
    assert failed["fold_2016_2018_strictly_exceeds_control"] is False


def test_control_canary_uses_absolute_1e_12_tolerance():
    row = _control_canary_row()
    v33.validate_control_canary(row)

    changed = copy.deepcopy(row)
    changed["full_calmar"] += 2e-12
    with pytest.raises(ValueError, match="control canary mismatch"):
        v33.validate_control_canary(changed)


def test_metadata_hashes_and_outputs_are_complete_and_refuse_overwrite(
    tmp_path: Path,
):
    plan = tmp_path / "plan.json"
    runtime = tmp_path / "runtime.npz"
    script = tmp_path / "script.py"
    candidate_source = tmp_path / "candidates.json"
    factor = tmp_path / "factor.py"
    for path, value in (
        (plan, b"plan"),
        (runtime, b"runtime"),
        (script, b"script"),
        (candidate_source, b"candidate"),
        (factor, b"factor"),
    ):
        path.write_bytes(value)

    metadata = v33.build_run_metadata(
        plan_path=plan,
        runtime_path=runtime,
        script_path=script,
        candidate_source_path=candidate_source,
        factor_source_paths={v33.NEW_FACTOR_NAME: factor},
        elapsed_seconds=1.25,
        loader_post_train_rows=7,
        runtime_panel_first_date="2008-01-02",
        runtime_panel_last_date=v33.TRAIN_LAST_DATE,
        rank_pool_column_count=2,
        runtime_column_count=3,
    )
    assert metadata["preregistered_plan_sha256"] == v33._sha256(plan)
    assert metadata["runtime_sha256"] == v33._sha256(runtime)
    assert metadata["research_script_sha256"] == v33._sha256(script)
    assert metadata["factor_source_sha256"][v33.NEW_FACTOR_NAME]["sha256"] == (
        v33._sha256(factor)
    )
    assert metadata["other_blend_weights_evaluated"] is False
    assert metadata["strategy_holdout_available_to_factor_or_worker"] is False

    output = tmp_path / "output"
    rows = [{"role": "control"}, {"role": "standalone"}, {"role": "blend"}]
    summary = {"status": "test"}
    dates = ["2018-12-27", "2018-12-28"]
    daily_returns = {
        "control": np.asarray([0.0, 1.0]),
        "standalone": np.asarray([1.0, 0.0]),
        "blend": np.asarray([0.25, 0.75]),
    }
    daily_exposures = {
        "control": np.asarray([0.5, 0.6]),
        "standalone": np.asarray([0.7, 0.8]),
        "blend": np.asarray([0.55, 0.65]),
    }
    v33.write_outputs(
        output,
        rows,
        summary,
        metadata,
        dates,
        daily_returns,
        daily_exposures,
    )
    assert {path.name for path in output.iterdir()} == {
        "all_results.json",
        "daily_returns.npz",
        "summary.json",
        "run_metadata.json",
    }
    with np.load(output / "daily_returns.npz") as saved:
        np.testing.assert_array_equal(saved["dates"], dates)
        np.testing.assert_allclose(saved["blend_daily_returns"], [0.25, 0.75])
        np.testing.assert_allclose(saved["blend_daily_exposures"], [0.55, 0.65])

    with pytest.raises(FileExistsError):
        v33.write_outputs(
            output,
            rows,
            summary,
            metadata,
            dates,
            daily_returns,
            daily_exposures,
        )
