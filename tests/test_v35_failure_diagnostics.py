from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pytest

import research_v35_failure_diagnostics as diagnostics


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_frozen_hashes_are_complete_lowercase_sha256_digests():
    digests = [
        *diagnostics.EXPECTED_AUTHORITATIVE_SHA256.values(),
        diagnostics.EXPECTED_AUTHORITATIVE_MANIFEST_SHA256,
        diagnostics.EXPECTED_PLAN_SHA256,
        diagnostics.EXPECTED_AMENDMENT_SHA256,
        diagnostics.EXPECTED_CANDIDATE_SOURCE_SHA256,
        diagnostics.EXPECTED_RUNTIME_SHA256,
        diagnostics.EXPECTED_AUTHORITATIVE_SCRIPT_SHA256,
        *diagnostics.EXPECTED_FACTOR_SOURCE_SHA256.values(),
        *diagnostics.EXPECTED_EXECUTION_SOURCE_SHA256.values(),
    ]

    assert set(diagnostics.EXPECTED_AUTHORITATIVE_SHA256) == set(
        diagnostics.AUTHORITATIVE_FILENAMES
    )
    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in digests)


def test_default_paths_and_authoritative_identities_are_pinned():
    assert diagnostics.DEFAULT_OUTPUT == Path(
        "results/strategy_opt_20260721/"
        "v35_prior_month_turnover_fixed_ablation_failure_diagnostics.json"
    )
    assert diagnostics._sha256(diagnostics.v35.PLAN_PATH) == (
        diagnostics.EXPECTED_PLAN_SHA256
    )
    assert diagnostics._sha256(diagnostics.v35.AMENDMENT_PATH) == (
        diagnostics.EXPECTED_AMENDMENT_SHA256
    )
    assert diagnostics._sha256(
        diagnostics.AUTHORITATIVE_RESULT_DIR / "manifest.json"
    ) == diagnostics.EXPECTED_AUTHORITATIVE_MANIFEST_SHA256
    assert diagnostics._sha256(
        Path("research_v35_prior_month_turnover_ablation.py")
    ) == diagnostics.EXPECTED_AUTHORITATIVE_SCRIPT_SHA256
    for name, digest in diagnostics.EXPECTED_AUTHORITATIVE_SHA256.items():
        assert diagnostics._sha256(
            diagnostics.AUTHORITATIVE_RESULT_DIR / name
        ) == digest


def test_manifest_hashes_every_artifact_before_parsing_json(tmp_path: Path):
    generated: dict[str, str] = {}
    names = set(diagnostics.AUTHORITATIVE_FILENAMES)
    for label in (diagnostics.v35.CONTROL_LABEL, diagnostics.v35.CANDIDATE_LABEL):
        names.update(
            f"{label}/{name}" for name in diagnostics.v35.ARM_REQUIRED_FILENAMES
        )
        names.add(f"{label}/backtest_report_20000101_000000.html")
    names.add("zz-last.bin")
    for name in sorted(names):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not valid JSON")
        generated[name] = diagnostics._sha256(path)
    generated["zz-last.bin"] = "0" * 64
    manifest = {
        "experiment": diagnostics.AUTHORITATIVE_EXPERIMENT,
        "status": "fixed_candidate_rejected_at_primary_gates",
        "input_hashes_verified_at_start_and_end": True,
        "generated_artifact_sha256": generated,
    }
    _write_json(tmp_path / "manifest.json", manifest)
    root_hashes = {
        name: diagnostics._sha256(tmp_path / name)
        for name in diagnostics.AUTHORITATIVE_FILENAMES
    }

    with pytest.raises(ValueError, match="artifact zz-last.bin SHA256"):
        diagnostics.load_authoritative_result(
            tmp_path,
            root_hashes,
            diagnostics._sha256(tmp_path / "manifest.json"),
        )


def test_default_inputs_cross_validate_as_rejected_training_only_result():
    validated, identities = diagnostics.validate_inputs(
        result_dir=diagnostics.AUTHORITATIVE_RESULT_DIR,
        plan_path=diagnostics.v35.PLAN_PATH,
        amendment_path=diagnostics.v35.AMENDMENT_PATH,
        candidate_source_path=diagnostics.v35.CANDIDATE_SOURCE_PATH,
        runtime_path=diagnostics.RUNTIME_PATH,
    )

    assert validated["authoritative"]["summary"]["status"] == (
        "fixed_candidate_rejected_at_primary_gates"
    )
    assert [row["role"] for row in validated["candidates"]] == [
        "control",
        "candidate",
    ]
    assert identities["authoritative_result"]["manifest"]["sha256"] == (
        diagnostics.EXPECTED_AUTHORITATIVE_MANIFEST_SHA256
    )
    assert len(
        identities["authoritative_result"]["generated_artifacts"]
    ) == 21


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

    class CompletedPriorMonthTurnoverStrict:
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
            [CompletedPriorMonthTurnoverStrict],
            [],
        )
    )

    assert discarded == 1
    assert seen == [diagnostics.TRAIN_LAST_DATE, diagnostics.TRAIN_LAST_DATE]
    assert raw.shape == (2, 2)
    assert score_keys == {diagnostics.v35.NEW_FACTOR_NAME}
    assert diagnostics.v35.NEW_FACTOR_NAME in arrays
    assert not np.shares_memory(data["open"], original_open)
    assert str(data["trade_dates"][-1]) == diagnostics.TRAIN_LAST_DATE


def test_exact_authoritative_series_audit_uses_zero_tolerance():
    expected = {
        "control_daily_returns": np.asarray([0.1, -0.2]),
        "candidate_daily_returns": np.asarray([0.2, -0.1]),
        "control_daily_exposures": np.asarray([0.4, 0.5]),
        "candidate_daily_exposures": np.asarray([0.3, 0.6]),
        "control_daily_selectable_candidates": np.asarray([100, 101]),
        "candidate_daily_selectable_candidates": np.asarray([99, 100]),
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
        "strict_candidate_mask_control_rank_control_timing": 101.0,
        "strict_candidate_mask_control_rank_recomputed_timing": 102.0,
        "candidate_rank_strict_mask_same_mask_timing": 110.0,
        "formal_candidate": 115.0,
    }

    result = diagnostics.decompose_terminal_navs(navs)
    identity = result["identity"]
    assert identity["direct_candidate_to_control_wealth_ratio"] == pytest.approx(
        1.15
    )
    assert identity["product_of_component_wealth_ratios"] == pytest.approx(1.15)
    assert identity["absolute_product_error"] <= 1e-12
    assert identity["absolute_log_sum_error"] <= 1e-12


def test_top30_uses_composite_rank_masks_and_reports_target_rebalance():
    factor = diagnostics.v35.NEW_FACTOR_NAME
    arrays = {
        "Base": np.asarray(
            [[0.9, 0.8, 0.7, 0.6], [0.6, 0.9, 0.8, 0.7]],
            dtype=np.float32,
        ),
        factor: np.asarray(
            [[0.1, 0.9, 0.8, 0.7], [0.9, 0.1, 0.8, 0.7]],
            dtype=np.float32,
        ),
        "_factor_valid_Base": np.ones((2, 4), dtype=bool),
        f"_factor_valid_{factor}": np.asarray(
            [[False, True, True, True], [True, True, True, True]], dtype=bool
        ),
        "Shared": np.ones((2, 4), dtype=bool),
    }
    base_config = {
        "weights": {"Base": 1.0},
        "filter_factors": {"Shared": True},
    }
    candidate_config = {
        "weights": {"Base": 1.0, factor: 0.1},
        "filter_factors": {"Shared": True},
    }
    candidates = [
        {"role": "control", "individual_config": base_config},
        {"role": "candidate", "individual_config": candidate_config},
    ]

    result = diagnostics.top30_diagnostics(
        arrays=arrays,
        score_keys={"Base", factor},
        candidates=candidates,
        dates=["2010-01-04", "2010-01-05"],
        date_indices=[0, 1],
        pool_columns=np.arange(4, dtype=np.intp),
        top_n=2,
    )

    assert result["stock_slot_denominator"] == 4
    assert result["control_top30_new_factor_valid"]["numerator"] == 3
    assert result["pure_rank_target_rebalance"]["transition_days"] == 1
    assert "LegalityChecker" in result["definition"]


def test_coverage_is_measured_inside_control_selectable_pool():
    factor = diagnostics.v35.NEW_FACTOR_NAME
    shape = (2, 3)
    arrays = {
        "Base": np.ones(shape, dtype=np.float32),
        factor: np.ones(shape, dtype=np.float32),
        "_factor_valid_Base": np.ones(shape, dtype=bool),
        f"_factor_valid_{factor}": np.asarray(
            [[True, False, True], [True, True, True]], dtype=bool
        ),
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
                "weights": {"Base": 1.0, factor: 0.1},
                "filter_factors": {"Shared": True},
            },
        },
    ]
    raw = np.asarray([[1.0, np.nan, 2.0], [1.0, 2.0, 3.0]])

    result = diagnostics.coverage_diagnostics(
        data={},
        arrays=arrays,
        score_keys={"Base", factor},
        candidates=candidates,
        date_indices=[0, 1],
        pool_columns=np.arange(3, dtype=np.intp),
        raw_new_factor=raw,
    )

    coverage = result["new_factor_coverage_in_control_selectable_pool"]
    assert coverage["denominator_control_selectable_stock_days"] == 6
    assert coverage["valid_control_selectable_stock_days"] == 5
    assert coverage["invalid_control_selectable_stock_days"] == 1


def test_rank_ic_is_strict_daily_next_open_spearman():
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
    assert result["definition"]["minimum_stock_sample_per_day"] == 3
    assert "trade legality" in result["definition"]["excluded"]


def test_raw_turnover_redundancy_uses_daily_spearman_against_all_base_ranks():
    shape = (3, 3)
    raw = np.tile(np.asarray([[1.0, 2.0, 3.0]]), (3, 1))
    arrays: dict[str, np.ndarray] = {
        "Shared": np.ones(shape, dtype=bool)
    }
    for index, name in enumerate(diagnostics.v35.BASE_FACTOR_NAMES):
        values = raw if index % 2 == 0 else raw[:, ::-1]
        arrays[name] = values.astype(np.float32)
        arrays[f"_factor_valid_{name}"] = np.ones(shape, dtype=bool)

    result = diagnostics.base_rank_redundancy_diagnostics(
        arrays=arrays,
        candidate_config={"filter_factors": {"Shared": True}},
        raw_new_factor=raw,
        dates=["2010-01-04", "2010-01-05", "2010-01-06"],
        date_indices=[0, 1, 2],
        pool_columns=np.arange(3, dtype=np.intp),
        minimum_sample=3,
    )

    assert set(result["by_base_factor"]) == set(
        diagnostics.v35.BASE_FACTOR_NAMES
    )
    assert result["by_base_factor"]["PreCloseMarketCap"][
        "mean_daily_spearman"
    ] == pytest.approx(1.0)
    assert result["by_base_factor"]["AmihudIlliquidityStrict"][
        "mean_daily_spearman"
    ] == pytest.approx(-1.0)
    assert "trade legality" in result["definition"]["excluded"]


def test_primary_gate_margins_recompute_the_authoritative_rejection():
    authoritative, _ = diagnostics.load_authoritative_result(
        diagnostics.AUTHORITATIVE_RESULT_DIR
    )
    result = diagnostics.primary_gate_diagnostics(authoritative)

    assert result["failed_gate_names"] == [
        "robust_score_strictly_exceeds_control",
        "worst_fold_calmar_at_least_1p5",
    ]
    margins = result["candidate_minus_gate_or_control_margins"]
    assert margins["worst_fold_calmar_minus_1p5"] == pytest.approx(
        -0.5350943377587511
    )
    assert margins["fold_2016_2018_minus_control"] == pytest.approx(
        0.00014524397663173705
    )


def test_post_failure_return_diagnostics_exclude_preregistered_secondary_audits():
    authoritative, _ = diagnostics.load_authoritative_result(
        diagnostics.AUTHORITATIVE_RESULT_DIR
    )
    result = diagnostics.authoritative_return_diagnostics(authoritative)

    assert set(result) == {
        "return_unit",
        "authoritative_metrics",
        "daily_return_correlation",
        "yearly_comparison",
    }
    source = Path(diagnostics.__file__).read_text(encoding="utf-8")
    assert "def rolling_756_relative_diagnostics" not in source
    assert '"rolling_756_candidate_relative_to_control"' not in source


def test_formal_numeric_canaries_pin_both_arms_and_known_candidate_counts():
    expected = diagnostics.EXPECTED_NUMERIC_CANARIES
    assert expected["control_formal_calmar"] == (3.087848127541097, 1e-12)
    assert expected["control_formal_terminal_nav"] == (
        44.18310168124901,
        1e-12,
    )
    assert expected["candidate_formal_calmar"] == (
        3.0856029262207474,
        1e-12,
    )
    assert expected["candidate_formal_worst_fold_calmar"] == (
        0.9649056622412489,
        1e-12,
    )
    assert expected["candidate_formal_minimum_selectable"] == (113.0, 0.0)
    assert expected["candidate_formal_trade_rows"] == (50_855.0, 0.0)


def test_numeric_canary_mismatch_is_fail_loud_and_keeps_actual_value():
    audit = diagnostics.evaluate_numeric_canaries(
        {"only": 2.0}, expected={"only": (1.0, 0.0)}
    )
    assert audit["checks"]["only"]["actual"] == 2.0
    assert audit["all_pass"] is False
    with pytest.raises(ValueError, match='"actual": 2.0'):
        diagnostics._assert_numeric_canaries(audit)


def test_formal_trade_rebalance_and_report_audit_is_exact():
    authoritative, _ = diagnostics.load_authoritative_result(
        diagnostics.AUTHORITATIVE_RESULT_DIR
    )
    result = diagnostics.execution_report_audit(
        authoritative, diagnostics.AUTHORITATIVE_RESULT_DIR
    )

    assert result["arms"]["control"]["trade_rows"] == 50_469
    assert result["arms"]["candidate"]["trade_rows"] == 50_855
    assert result["candidate_minus_control"]["trade_rows"] == 386
    assert result["candidate_minus_control"][
        "executed_rebalance_days_with_buys_and_sells"
    ] == 19
    assert result["root_exact_open_audit_all_pass"] is True


def test_existing_output_is_rejected_before_any_input_is_opened(tmp_path: Path):
    output = tmp_path / "diagnostic.json"
    output.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        diagnostics.run(
            output,
            result_dir=tmp_path / "missing-result",
            plan_path=tmp_path / "missing-plan",
            amendment_path=tmp_path / "missing-amendment",
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
        redundancy={},
        returns={"authoritative_metrics": {}},
        gates={"fold_pattern": {"description": "training fold pattern"}},
        execution={},
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


def test_source_contains_no_sealed_period_or_ga_execution_path():
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
    forbidden_path_tokens = (
        "validation",
        "holdout",
        "test_period",
        "test/",
        "oos",
    )

    assert not any(
        token in literal.lower()
        for literal in path_like_literals
        for token in forbidden_path_tokens
    )
    assert "research_holdout" not in source
    assert "holdout_diagnostics.json" not in source
    assert "testback.run_ga" not in source


def test_diagnostic_source_is_not_the_authoritative_script_canary():
    source = Path(diagnostics.__file__).read_bytes()
    assert hashlib.sha256(source).hexdigest() != (
        diagnostics.EXPECTED_AUTHORITATIVE_SCRIPT_SHA256
    )
