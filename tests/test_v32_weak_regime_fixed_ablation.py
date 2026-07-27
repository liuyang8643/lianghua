import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from research_v32_weak_regime_fixed_ablation import (
    ALL_FACTOR_NAMES,
    BASE_FACTOR_NAMES,
    CANARY_ABSOLUTE_TOLERANCE,
    CONTROL_CANARY,
    CONTROL_LABEL,
    CONTROL_WEIGHTS,
    LONG_VOL_LABEL,
    NEW_FACTOR_NAMES,
    VOLUME_CONTRACTION_LABEL,
    annotate_primary_decision,
    build_fixed_candidates,
    build_run_metadata,
    candidate_filter_masks,
    load_and_validate_plan,
    load_control_source,
    physically_truncate_runtime,
    primary_gate_checks,
    run,
    validate_control_canary,
    write_outputs,
)


def _metric_row(
    *,
    role,
    label,
    full,
    folds,
    exposure,
    robust=None,
):
    worst = min(folds.values())
    if robust is None:
        robust = 0.5 * full + 0.5 * worst
    return {
        "candidate_index": 0 if role == "control" else 1,
        "role": role,
        "label": label,
        "individual_config": {},
        "full_calmar": full,
        "fold_calmars": dict(folds),
        "worst_fold_calmar": worst,
        "robust_score": robust,
        "fitness": robust,
        "average_exposure": exposure,
        "exposure_constraint_passed": exposure >= 0.45,
        "annualized": 50.0,
        "max_drawdown": -20.0,
        "sharpe": 2.0,
        "total_return": 1000.0,
    }


def _control_row():
    return _metric_row(
        role="control",
        label=CONTROL_LABEL,
        full=CONTROL_CANARY["full_calmar"],
        folds=CONTROL_CANARY["fold_calmars"],
        exposure=CONTROL_CANARY["average_exposure"],
        robust=CONTROL_CANARY["robust_score"],
    )


def test_plan_and_candidate_semantics_are_exactly_three_fixed_arms():
    plan = load_and_validate_plan()
    source = load_control_source()
    candidates = build_fixed_candidates(plan, source)

    assert [row["label"] for row in candidates] == [
        CONTROL_LABEL,
        LONG_VOL_LABEL,
        VOLUME_CONTRACTION_LABEL,
    ]
    assert [row["role"] for row in candidates] == ["control", "arm", "arm"]
    assert [row["candidate_index"] for row in candidates] == [0, 1, 2]

    source_semantics = copy.deepcopy(source["individual_config"])
    source_semantics.pop("weights")
    expected_new = [(0.0, 0.0), (0.1, 0.0), (0.0, 0.1)]
    for row, pair in zip(candidates, expected_new):
        config = copy.deepcopy(row["individual_config"])
        weights = config.pop("weights")
        assert config == source_semantics
        assert tuple(weights) == ALL_FACTOR_NAMES
        assert {name: weights[name] for name in BASE_FACTOR_NAMES} == CONTROL_WEIGHTS
        assert tuple(weights[name] for name in NEW_FACTOR_NAMES) == pair
        assert not all(weights[name] != 0.0 for name in NEW_FACTOR_NAMES)


def test_plan_rejects_a_combined_or_changed_arm(tmp_path):
    payload = json.loads(
        Path(
            "results/strategy_opt_20260721/"
            "v32_weak_regime_fixed_ablation_preregistered_plan.json"
        ).read_text(encoding="utf-8")
    )
    payload["arms"][0]["change_from_control"][
        "VolumeContraction5v15Strict"
    ] = 0.1
    changed = tmp_path / "changed_plan.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="arm identity|combined"):
        load_and_validate_plan(changed)


def test_zero_weight_factor_validity_does_not_filter_other_arms():
    candidates = build_fixed_candidates(
        load_and_validate_plan(),
        load_control_source(),
    )
    shape = (2, 3)
    arrays = {}
    for name in BASE_FACTOR_NAMES:
        arrays[f"_factor_valid_{name}"] = np.ones(shape, dtype=bool)
    long_valid = np.asarray(
        [[True, False, True], [True, True, False]],
        dtype=bool,
    )
    volume_valid = np.asarray(
        [[False, False, False], [False, True, True]],
        dtype=bool,
    )
    arrays["_factor_valid_LongHorizonVol36MStrict"] = long_valid
    arrays["_factor_valid_VolumeContraction5v15Strict"] = volume_valid

    control_masks = candidate_filter_masks(
        arrays,
        ALL_FACTOR_NAMES,
        candidates[0]["individual_config"],
    )
    np.testing.assert_array_equal(
        control_masks["_active_factor_intersection"],
        np.ones(shape, dtype=bool),
    )
    long_masks = candidate_filter_masks(
        arrays,
        ALL_FACTOR_NAMES,
        candidates[1]["individual_config"],
    )
    np.testing.assert_array_equal(
        long_masks["_active_factor_intersection"],
        long_valid,
    )
    volume_masks = candidate_filter_masks(
        arrays,
        ALL_FACTOR_NAMES,
        candidates[2]["individual_config"],
    )
    np.testing.assert_array_equal(
        volume_masks["_active_factor_intersection"],
        volume_valid,
    )


def test_runtime_holdout_rows_are_physically_copied_out():
    dates = np.asarray(
        ["2018-12-27", "2018-12-28", "2019-01-02"],
        dtype="datetime64[D]",
    )
    open_panel = np.arange(6, dtype=np.float64).reshape(3, 2)
    data = {
        "trade_dates": dates,
        "stock_codes": np.asarray(["000001.SZ", "600000.SH"]),
        "open": open_panel,
        "close": open_panel + 1.0,
        "st_mask": np.zeros((3, 2), dtype=bool),
        "issue_price": np.asarray([1.0, 2.0]),
    }
    truncated, post_rows = physically_truncate_runtime(data)
    assert post_rows == 1
    assert truncated["trade_dates"].tolist() == dates[:2].tolist()
    assert truncated["open"].shape == (2, 2)
    assert not np.shares_memory(truncated["trade_dates"], dates)
    assert not np.shares_memory(truncated["open"], open_panel)
    truncated["open"][0, 0] = -999.0
    assert data["open"][0, 0] != -999.0


def test_runtime_truncation_requires_the_fixed_training_end():
    data = {
        "trade_dates": np.asarray(["2018-12-27"], dtype="datetime64[D]"),
        "stock_codes": np.asarray(["000001.SZ"]),
        "open": np.ones((1, 1)),
    }
    with pytest.raises(ValueError, match="final training date"):
        physically_truncate_runtime(data)


def test_control_canary_checks_full_folds_and_exposure_at_1e12():
    row = _control_row()
    validate_control_canary(row)
    assert CANARY_ABSOLUTE_TOLERANCE == 1e-12

    bad = copy.deepcopy(row)
    bad["fold_calmars"]["2016-2018"] += 2e-12
    with pytest.raises(ValueError, match="canary mismatch"):
        validate_control_canary(bad)

    bad = copy.deepcopy(row)
    bad["average_exposure"] += 2e-12
    with pytest.raises(ValueError, match="canary mismatch"):
        validate_control_canary(bad)


def test_primary_gates_require_every_absolute_and_control_relative_condition():
    control = _metric_row(
        role="control",
        label=CONTROL_LABEL,
        full=2.6,
        folds={
            "2010-2012": 1.6,
            "2013-2015": 2.0,
            "2016-2018": 1.6,
        },
        exposure=0.5,
        robust=2.1,
    )
    passing = _metric_row(
        role="arm",
        label=LONG_VOL_LABEL,
        full=2.7,
        folds={
            "2010-2012": 1.7,
            "2013-2015": 2.0,
            "2016-2018": 1.7,
        },
        exposure=0.5,
        robust=2.2,
    )
    checks = primary_gate_checks(passing, control)
    assert checks and all(checks.values())

    equal_weak_fold = copy.deepcopy(passing)
    equal_weak_fold["fold_calmars"]["2016-2018"] = 1.6
    assert not primary_gate_checks(equal_weak_fold, control)[
        "fold_2016_2018_strictly_exceeds_control"
    ]

    equal_robust = copy.deepcopy(passing)
    equal_robust["robust_score"] = control["robust_score"]
    assert not primary_gate_checks(equal_robust, control)[
        "robust_score_strictly_exceeds_control"
    ]

    nonfinite = copy.deepcopy(passing)
    nonfinite["sharpe"] = np.nan
    failed = primary_gate_checks(nonfinite, control)
    assert not failed["all_metrics_finite"]
    assert not all(failed.values())


def test_failed_arms_are_never_selected_relatively():
    control = _control_row()
    almost = _metric_row(
        role="arm",
        label=LONG_VOL_LABEL,
        full=3.2,
        folds={
            "2010-2012": 1.6,
            "2013-2015": 8.0,
            "2016-2018": 1.49,
        },
        exposure=0.6,
    )
    weak = _metric_row(
        role="arm",
        label=VOLUME_CONTRACTION_LABEL,
        full=2.0,
        folds={
            "2010-2012": 0.8,
            "2013-2015": 2.0,
            "2016-2018": 0.7,
        },
        exposure=0.6,
    )
    almost["candidate_index"] = 1
    weak["candidate_index"] = 2
    annotated, decision = annotate_primary_decision([control, almost, weak])
    assert decision == {
        "status": "no_arm_passed_all_primary_gates",
        "primary_passing_arm_count": 0,
        "primary_passing_arms": [],
        "relative_best_failure_selected": False,
        "combined_arm_evaluated": False,
        "configuration_freeze_allowed": False,
        "secondary_audit_required_for": [],
    }
    assert annotated[1]["passes_all_primary_gates"] is False
    assert annotated[2]["passes_all_primary_gates"] is False


def test_metadata_records_identity_hashes_and_sealed_contract(tmp_path):
    paths = {}
    for name in ("plan", "runtime", "script", "candidates", "long", "volume"):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode("ascii"))
        paths[name] = path
    metadata = build_run_metadata(
        plan_path=paths["plan"],
        runtime_path=paths["runtime"],
        script_path=paths["script"],
        candidate_source_path=paths["candidates"],
        factor_source_paths={
            "LongHorizonVol36MStrict": paths["long"],
            "VolumeContraction5v15Strict": paths["volume"],
        },
        workers_requested=8,
        worker_processes=3,
        shared_memory_bytes=123,
        elapsed_seconds=4.5,
        loader_post_train_rows=7,
        runtime_panel_first_date="2006-04-03",
        runtime_panel_last_date="2018-12-28",
        rank_pool_column_count=4737,
        runtime_column_count=5350,
    )
    for key, name in (
        ("preregistered_plan_sha256", "plan"),
        ("runtime_sha256", "runtime"),
        ("research_script_sha256", "script"),
    ):
        assert metadata[key] == hashlib.sha256(
            paths[name].read_bytes()
        ).hexdigest()
    assert metadata["runtime_panel_last_date_after_copy"] == "2018-12-28"
    assert metadata["runtime_loader_returned_post_train_rows"] == 7
    assert metadata["strategy_holdout_available_to_factor_or_worker"] is False
    assert metadata["strategy_holdout_evaluated"] is False
    assert metadata["zero_weight_factor_validity_affects_candidate_mask"] is False
    assert metadata["combined_arm_evaluated"] is False
    assert metadata["workers_requested"] == 8
    assert metadata["worker_processes"] == 3


def test_output_is_complete_and_refuses_overwrite(tmp_path):
    output = tmp_path / "result"
    rows = [{"candidate_index": value} for value in range(3)]
    summary = {"status": "x"}
    metadata = {"sealed_holdout": True}
    write_outputs(output, rows, summary, metadata)
    assert {path.name for path in output.iterdir()} == {
        "all_results.jsonl",
        "summary.json",
        "run_metadata.json",
    }
    loaded = [
        json.loads(line)
        for line in (output / "all_results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert loaded == rows
    with pytest.raises(FileExistsError):
        write_outputs(output, rows, summary, metadata)


def test_run_rejects_existing_output_before_loading_runtime(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        run(output, workers=1)


def test_run_rejects_nonpositive_workers_before_loading_runtime(tmp_path):
    with pytest.raises(ValueError, match="workers"):
        run(tmp_path / "new", workers=0)
