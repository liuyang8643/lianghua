from __future__ import annotations

import ast
import json
from pathlib import Path
import re

import numpy as np
import pytest

import research_v34_failure_diagnostics as diagnostics


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _identity(path: Path) -> dict:
    return {"path": str(path), "sha256": diagnostics._sha256(path)}


def test_frozen_input_hashes_are_complete_lowercase_sha256_digests():
    digests = [
        *diagnostics.EXPECTED_AUTHORITATIVE_SHA256.values(),
        diagnostics.EXPECTED_PLAN_SHA256,
        diagnostics.EXPECTED_CANDIDATE_SOURCE_SHA256,
        diagnostics.EXPECTED_RUNTIME_SHA256,
        diagnostics.EXPECTED_ADAPTER_SHA256,
        diagnostics.EXPECTED_REPORT_GENERATOR_SHA256,
        diagnostics.EXPECTED_TRAIN_REPORT_MANIFEST_SHA256,
        *diagnostics.EXPECTED_FACTOR_SOURCE_SHA256.values(),
    ]

    assert set(diagnostics.EXPECTED_AUTHORITATIVE_SHA256) == set(
        diagnostics.AUTHORITATIVE_FILENAMES
    )
    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in digests)


def test_default_plan_manifest_and_authoritative_files_match_frozen_identity():
    assert diagnostics.DEFAULT_OUTPUT == Path(
        "results/strategy_opt_20260721/"
        "v34_same_calendar_month_return_fixed_ablation_failure_diagnostics.json"
    )
    assert diagnostics._sha256(diagnostics.v34.PLAN_PATH) == (
        diagnostics.EXPECTED_PLAN_SHA256
    )
    assert diagnostics._sha256(
        diagnostics.TRAIN_REPORT_DIR / "manifest.json"
    ) == diagnostics.EXPECTED_TRAIN_REPORT_MANIFEST_SHA256
    for name, digest in diagnostics.EXPECTED_AUTHORITATIVE_SHA256.items():
        assert diagnostics._sha256(
            diagnostics.AUTHORITATIVE_RESULT_DIR / name
        ) == digest


def test_authoritative_loader_hashes_every_file_before_parsing(tmp_path: Path):
    expected = {}
    for name in diagnostics.AUTHORITATIVE_FILENAMES:
        path = tmp_path / name
        path.write_bytes(name.encode("ascii"))
        expected[name] = diagnostics._sha256(path)
    expected["summary.json"] = "0" * 64

    with pytest.raises(ValueError, match="authoritative summary.json SHA256"):
        diagnostics.load_authoritative_result(tmp_path, expected)


def test_default_frozen_inputs_cross_validate_without_opening_any_other_period():
    validated, identities = diagnostics.validate_inputs(
        result_dir=diagnostics.AUTHORITATIVE_RESULT_DIR,
        report_dir=diagnostics.TRAIN_REPORT_DIR,
        plan_path=diagnostics.v34.PLAN_PATH,
        candidate_source_path=diagnostics.v34.CANDIDATE_SOURCE_PATH,
        runtime_path=diagnostics.RUNTIME_PATH,
    )

    assert validated["authoritative"]["summary"]["status"] == (
        "fixed_candidate_rejected_at_primary_gates"
    )
    assert [row["role"] for row in validated["candidates"]] == [
        "control",
        "candidate",
    ]
    assert identities["formal_training_reports"]["manifest"]["sha256"] == (
        diagnostics.EXPECTED_TRAIN_REPORT_MANIFEST_SHA256
    )


def test_formal_training_manifest_cross_checks_and_hashes_all_artifacts(
    tmp_path: Path,
):
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    exact_flags = {
        "natural_metrics_exactly_match_authoritative_v34": True,
        "daily_returns_exactly_match_authoritative_v34": True,
        "daily_exposures_exactly_match_authoritative_v34": True,
        "daily_selectable_counts_exactly_match_authoritative_v34": True,
    }
    report_summary = {
        "selection_scope": "training_only_2010_2018",
        "authoritative_status": "fixed_candidate_rejected_at_primary_gates",
        "reports_are_diagnostic_and_cannot_change_rejected_selection": True,
        "runtime_physically_truncated_before_factor_calculation": True,
        "relative_best_failure_selected": False,
        "configuration_frozen": False,
        "strategy_holdout_available_to_factor_or_worker": False,
        "strategy_holdout_evaluated": False,
        "real_arm_reports": [
            {"role": "control", **exact_flags},
            {"role": "candidate", **exact_flags},
        ],
    }
    _write_json(report_dir / "summary.json", report_summary)
    nested = report_dir / "control"
    nested.mkdir()
    (nested / "trades.json").write_text("training only\n", encoding="utf-8")

    sources = {}
    for name in (
        "preregistered_plan",
        "candidate_source",
        "runtime",
        "authoritative_adapter",
        "report_generator",
    ):
        path = tmp_path / f"{name}.source"
        path.write_text(name, encoding="utf-8")
        sources[name] = _identity(path)
    factor_path = tmp_path / "factor.py"
    factor_path.write_text("factor", encoding="utf-8")
    factor_identities = {"Factor": _identity(factor_path)}
    authoritative_identities = {}
    for name in diagnostics.AUTHORITATIVE_FILENAMES:
        path = tmp_path / f"authoritative-{name}"
        path.write_text(name, encoding="utf-8")
        authoritative_identities[name] = _identity(path)

    manifest = {
        "experiment": (
            "v34_same_calendar_month_return_fixed_ablation_train_reports"
        ),
        "source_runtime_plan_adapter_factors_and_authoritative_result_sha256": {
            **sources,
            "factor_sources": factor_identities,
            "authoritative_result": authoritative_identities,
        },
        "generated_artifact_sha256": {
            "summary.json": diagnostics._sha256(report_dir / "summary.json"),
            "control/trades.json": diagnostics._sha256(
                nested / "trades.json"
            ),
        },
    }
    _write_json(report_dir / "manifest.json", manifest)

    identities, loaded_summary = diagnostics.validate_training_report_manifest(
        report_dir,
        authoritative_identities=authoritative_identities,
        plan_identity=sources["preregistered_plan"],
        candidate_source_identity=sources["candidate_source"],
        runtime_identity=sources["runtime"],
        adapter_identity=sources["authoritative_adapter"],
        factor_identities=factor_identities,
        report_generator_identity=sources["report_generator"],
        expected_manifest_sha256=diagnostics._sha256(
            report_dir / "manifest.json"
        ),
    )

    assert loaded_summary == report_summary
    assert set(identities["generated_artifacts"]) == {
        "summary.json",
        "control/trades.json",
    }
    (nested / "trades.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact control/trades.json SHA256"):
        diagnostics.validate_training_report_manifest(
            report_dir,
            authoritative_identities=authoritative_identities,
            plan_identity=sources["preregistered_plan"],
            candidate_source_identity=sources["candidate_source"],
            runtime_identity=sources["runtime"],
            adapter_identity=sources["authoritative_adapter"],
            factor_identities=factor_identities,
            report_generator_identity=sources["report_generator"],
            expected_manifest_sha256=diagnostics._sha256(
                report_dir / "manifest.json"
            ),
        )


def test_training_date_seal_is_exact_and_rejects_post_train_rows():
    assert diagnostics.TRAIN_FIRST_DATE == "2010-01-04"
    assert diagnostics.TRAIN_LAST_DATE == "2018-12-28"
    assert diagnostics.EXPECTED_TRAIN_DAYS == 2187
    series = diagnostics._load_daily_series(
        diagnostics.AUTHORITATIVE_RESULT_DIR / "daily_returns.npz"
    )
    valid = list(series["dates"])
    diagnostics._assert_strict_training_dates(valid)

    with pytest.raises(ValueError, match="sealed to the training period"):
        diagnostics._assert_strict_training_dates(valid[:-1] + ["2019-01-02"])
    with pytest.raises(ValueError, match="strictly increasing"):
        diagnostics._assert_strict_training_dates(
            valid[:100] + [valid[99]] + valid[101:]
        )


def test_factor_calls_happen_only_after_physical_training_truncation():
    seen: list[str] = []

    class SameCalendarMonthReturn3YStrict:
        hist_days = 1

        def calc_batch(self, panel):
            dates = np.asarray(panel["trade_dates"], dtype="datetime64[D]")
            seen.append(str(dates[-1]))
            return np.ones(
                (len(dates), len(panel["stock_codes"])), dtype=np.float32
            )

    loaded = {
        "trade_dates": np.asarray(
            ["2018-12-27", "2018-12-28", "2019-01-02"],
            dtype="datetime64[D]",
        ),
        "stock_codes": np.asarray(["000001", "600001"]),
        "open": np.ones((3, 2), dtype=np.float64),
    }
    original_open = loaded["open"]
    data, arrays, score_keys, raw, discarded = (
        diagnostics.compute_arrays_after_physical_truncation(
            loaded,
            [SameCalendarMonthReturn3YStrict],
            [],
        )
    )

    assert discarded == 1
    assert seen == [diagnostics.TRAIN_LAST_DATE, diagnostics.TRAIN_LAST_DATE]
    assert raw.shape == (2, 2)
    assert score_keys == {diagnostics.v34.NEW_FACTOR_NAME}
    assert diagnostics.v34.NEW_FACTOR_NAME in arrays
    assert not np.shares_memory(data["open"], original_open)
    assert str(data["trade_dates"][-1]) == diagnostics.TRAIN_LAST_DATE


def test_exact_authoritative_series_audit_uses_zero_tolerance():
    expected = {
        "control_daily_returns": np.asarray([0.1, -0.2]),
        "candidate_daily_returns": np.asarray([0.2, -0.1]),
        "control_daily_exposures": np.asarray([0.4, 0.5]),
        "candidate_daily_exposures": np.asarray([0.3, 0.6]),
        "control_daily_selectable_candidates": np.asarray([100, 101]),
        "candidate_daily_selectable_candidates": np.asarray([50, 51]),
    }
    paths = {
        "control": {
            "daily_returns": expected["control_daily_returns"].copy(),
            "daily_exposures": expected["control_daily_exposures"].copy(),
            "daily_selectable": expected[
                "control_daily_selectable_candidates"
            ].copy(),
        },
        "formal_candidate": {
            "daily_returns": expected["candidate_daily_returns"].copy(),
            "daily_exposures": expected["candidate_daily_exposures"].copy(),
            "daily_selectable": expected[
                "candidate_daily_selectable_candidates"
            ].copy(),
        },
    }

    audit = diagnostics.verify_authoritative_daily_series(paths, expected)
    assert all(row["exact"] for row in audit.values())
    assert all(row["absolute_tolerance"] == 0.0 for row in audit.values())

    paths["formal_candidate"]["daily_returns"][1] += 1e-15
    with pytest.raises(ValueError, match="not byte-value exact"):
        diagnostics.verify_authoritative_daily_series(paths, expected)


def test_terminal_wealth_decomposition_closes_multiplicatively_and_in_logs():
    navs = {
        "control": 100.0,
        "strict_candidate_mask_control_rank_control_timing": 80.0,
        "strict_candidate_mask_control_rank_recomputed_timing": 72.0,
        "candidate_rank_strict_mask_same_mask_timing": 75.6,
        "formal_candidate": 79.38,
    }

    result = diagnostics.decompose_terminal_navs(navs)
    identity = result["identity"]
    assert identity["direct_candidate_to_control_wealth_ratio"] == pytest.approx(
        0.7938
    )
    assert identity["product_of_component_wealth_ratios"] == pytest.approx(
        0.7938
    )
    assert identity["absolute_product_error"] <= 1e-12
    assert identity["absolute_log_sum_error"] <= 1e-12
    assert [row["wealth_effect_pct"] for row in result["components"]] == (
        pytest.approx([-20.0, -10.0, 5.0, 5.0])
    )


def test_top30_definition_uses_composite_rank_masks_without_trade_state():
    arrays = {
        "Base": np.asarray([[0.9, 0.8, 0.7, 0.6]], dtype=np.float32),
        diagnostics.v34.NEW_FACTOR_NAME: np.asarray(
            [[0.1, 0.9, 0.8, 0.7]], dtype=np.float32
        ),
        "_factor_valid_Base": np.ones((1, 4), dtype=bool),
        f"_factor_valid_{diagnostics.v34.NEW_FACTOR_NAME}": np.asarray(
            [[False, True, True, True]], dtype=bool
        ),
        "Shared": np.ones((1, 4), dtype=bool),
    }
    base_config = {
        "weights": {"Base": 1.0},
        "filter_factors": {"Shared": True},
    }
    candidate_config = {
        "weights": {"Base": 1.0, diagnostics.v34.NEW_FACTOR_NAME: 0.1},
        "filter_factors": {"Shared": True},
    }
    candidates = [
        {"role": "control", "individual_config": base_config},
        {"role": "candidate", "individual_config": candidate_config},
    ]

    result = diagnostics.top30_diagnostics(
        arrays=arrays,
        score_keys={"Base", diagnostics.v34.NEW_FACTOR_NAME},
        candidates=candidates,
        dates=["2010-01-04"],
        date_indices=[0],
        pool_columns=np.arange(4, dtype=np.intp),
        top_n=2,
    )

    assert result["stock_slot_denominator"] == 2
    assert result["control_top30_new_factor_valid"]["numerator"] == 1
    assert result["control_candidate_top30_overlap"]["intersection_slot_numerator"] == 1
    assert "LegalityChecker" in result["definition"]


def test_coverage_exclusion_reasons_separate_whole_and_partial_lag_months():
    month_count = 40
    dates = []
    for offset in range(month_count):
        month_start = (np.datetime64("2007-01", "M") + offset).astype(
            "datetime64[D]"
        )
        dates.extend((month_start, month_start + 1))
    trade_dates = np.asarray(dates, dtype="datetime64[D]")
    shape = (len(trade_dates), 2)
    close = np.ones(shape, dtype=np.float64)
    pre_close = np.ones(shape, dtype=np.float64)
    target_month = 37
    target_rows = np.asarray([target_month * 2, target_month * 2 + 1])
    whole_lag_rows = np.asarray([(target_month - 12) * 2, (target_month - 12) * 2 + 1])
    partial_lag_row = (target_month - 24) * 2
    close[whole_lag_rows, 0] = np.nan
    close[partial_lag_row, 1] = np.nan

    factor_name = diagnostics.v34.NEW_FACTOR_NAME
    factor_valid = np.ones(shape, dtype=bool)
    factor_valid[target_rows] = False
    raw = np.ones(shape, dtype=np.float32)
    raw[target_rows] = np.nan
    arrays = {
        "Base": np.ones(shape, dtype=np.float32),
        factor_name: np.ones(shape, dtype=np.float32),
        "_factor_valid_Base": np.ones(shape, dtype=bool),
        f"_factor_valid_{factor_name}": factor_valid,
        "Shared": np.ones(shape, dtype=bool),
    }
    candidates = [
        {
            "role": "control",
            "individual_config": {
                "weights": {"Base": 1.0},
                "filter_factors": {"Shared": True},
            },
        },
        {
            "role": "candidate",
            "individual_config": {
                "weights": {"Base": 1.0, factor_name: 0.1},
                "filter_factors": {"Shared": True},
            },
        },
    ]
    data = {
        "trade_dates": trade_dates,
        "close": close,
        "preClose": pre_close,
        "open": np.full(shape, 5.0),
        "st_mask": np.zeros(shape, dtype=bool),
    }

    result = diagnostics.coverage_diagnostics(
        data=data,
        arrays=arrays,
        score_keys={"Base", factor_name},
        candidates=candidates,
        date_indices=target_rows.tolist(),
        pool_columns=np.arange(2, dtype=np.intp),
        raw_new_factor=raw,
    )

    reasons = result["strict_invalidity_reasons"]
    assert reasons["excluded_control_selectable_stock_days"] == 4
    assert reasons["exclusive_counts"]["missing_whole_lag_month"] == 2
    assert reasons["exclusive_counts"][
        "partial_or_gapped_or_invalid_lag_month"
    ] == 2
    assert reasons["exclusive_counts_sum"] == 4


def test_rank_ic_definition_uses_next_open_spearman_and_ddof_one_t_stat():
    factor_name = diagnostics.v34.NEW_FACTOR_NAME
    raw = np.tile(np.asarray([[1.0, 2.0, 3.0]]), (4, 1))
    open_ = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [1.01, 1.02, 1.03],
            [1.0201, 1.0404, 1.0609],
            [1.030301, 1.061208, 1.092727],
        ]
    )
    arrays = {"Shared": np.ones((4, 3), dtype=bool)}
    result = diagnostics.rank_ic_diagnostics(
        data={"open": open_},
        arrays=arrays,
        candidate_config={"filter_factors": {"Shared": True}},
        raw_new_factor=raw,
        dates=["2010-01-04", "2010-01-05", "2010-01-06", "2010-01-07"],
        date_indices=[0, 1, 2, 3],
        pool_columns=np.arange(3, dtype=np.intp),
        minimum_sample=3,
    )

    assert result["valid_day_count"] == 3
    assert result["stock_day_sample_count"] == 9
    assert result["daily_rank_ic_mean"] == pytest.approx(1.0)
    assert result["daily_rank_ic_sample_std_ddof_1"] == pytest.approx(0.0)
    assert result["daily_rank_ic_t_stat"] is None
    assert result["definition"]["outcome"] == "open[T+1] / open[T] - 1"
    assert result["reference_disambiguation"][
        "not_comparable_to_current_daily_next_open_definition"
    ] is True
    assert result["reference_disambiguation"][
        "current_definition_changed_to_match_old_reference"
    ] is False


def test_daily_next_open_ic_canary_excludes_the_legacy_monthly_reference():
    expected, tolerance = diagnostics.EXPECTED_NUMERIC_CANARIES[
        "daily_next_open_spearman_rank_ic_mean"
    ]
    assert expected == 0.0010942140536300852
    assert tolerance == 1e-12
    assert expected != pytest.approx(0.0101, abs=1e-4)


def test_numeric_canary_mismatch_is_fail_loud_and_keeps_actual_value():
    audit = diagnostics.evaluate_numeric_canaries(
        {"only": 2.0}, expected={"only": (1.0, 0.0)}
    )
    assert audit["checks"]["only"]["actual"] == 2.0
    assert audit["all_pass"] is False
    with pytest.raises(ValueError, match='"actual": 2.0'):
        diagnostics._assert_numeric_canaries(audit)


def test_existing_output_is_rejected_before_any_input_is_opened(tmp_path: Path):
    output = tmp_path / "diagnostic.json"
    output.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        diagnostics.run(
            output,
            result_dir=tmp_path / "missing-result",
            report_dir=tmp_path / "missing-report",
            plan_path=tmp_path / "missing-plan",
            candidate_source_path=tmp_path / "missing-candidate-source",
            runtime_path=tmp_path / "missing-runtime",
        )
    assert output.read_text(encoding="utf-8") == "keep"


def test_output_contract_forbids_tuning_freeze_and_sealed_period_access():
    payload = diagnostics.build_output_payload(
        validated={
            "authoritative": {
                "summary": {
                    "status": "fixed_candidate_rejected_at_primary_gates"
                }
            }
        },
        identities={"runtime": {"path": "runtime", "sha256": "abc"}},
        exact_series_audit={"control": {"exact": True}},
        decomposition={"paths": {}},
        coverage={},
        top30={},
        rank_ic={},
        returns={"authoritative_metrics": {}, "yearly_comparison": []},
        canary_audit={"all_pass": True},
        post_train_rows_discarded=7,
        script_identity={"path": "script", "sha256": "def"},
    )

    contract = payload["diagnostic_contract"]
    assert contract["parameter_search_or_micro_tuning_performed"] is False
    assert contract["configuration_freeze_allowed"] is False
    assert contract["configuration_frozen"] is False
    assert contract["validation_test_holdout_loaded_or_evaluated"] is False
    assert contract["runtime_post_train_rows_discarded_before_factors"] == 7
    assert payload["why_no_parameter_tuning_or_freeze"][
        "post_failure_parameter_adjustment_would_be_a_new_experiment"
    ] is True


def test_source_contains_no_path_to_validation_test_or_legacy_holdout_data():
    source_path = Path(diagnostics.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    path_like_literals = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and ("/" in node.value or "\\" in node.value)
    ]
    forbidden = ("validation", "holdout", "legacy", "test_period", "test/")

    assert not any(
        token in literal.lower()
        for literal in path_like_literals
        for token in forbidden
    )
    assert "research_holdout" not in source
    assert "holdout_diagnostics.json" not in source
