import json
from pathlib import Path

import research_generate_v28_candidates as generator


PLAN_PATH = (
    Path(__file__).resolve().parents[1]
    / "results/strategy_opt_20260721/v28_two_arm_preregistered_plan.json"
)


def test_timing_grid_has_only_preregistered_coupled_dimensions():
    configs = generator.build_timing_candidates()

    assert len(configs) == 24
    assert len({item["label"] for item in configs}) == 24
    baseline = json.loads(generator.BASELINE_PATH.read_text(encoding="utf-8"))[
        "individual_config"
    ]
    baseline_hits = [
        item
        for item in configs
        if item["individual_config"] == baseline
    ]
    assert len(baseline_hits) == 1
    assert baseline_hits[0]["label"] == (
        "threshold=legacy;scale=1;strategy_weight=0.8"
    )


def test_weight_grid_is_complete_and_contains_v25_baseline_once():
    configs = generator.build_weight_candidates()

    assert len(configs) == 648
    assert len({item["label"] for item in configs}) == 648
    baseline = json.loads(generator.BASELINE_PATH.read_text(encoding="utf-8"))[
        "individual_config"
    ]
    baseline_hits = [
        item
        for item in configs
        if item["individual_config"] == baseline
    ]
    assert len(baseline_hits) == 1
    assert baseline_hits[0]["label"] == (
        "buy_n=40;amihud=0.2;reversal=0.1;volume_cv=0.1"
    )


def test_v28_plan_is_preregistered_and_forbids_holdout_during_selection():
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))

    assert plan["status"] == "preregistered_before_candidate_evaluation"
    assert plan["arms"]["completed_timing"]["candidate_count"] == 24
    assert plan["arms"]["strict_factor_weights"]["candidate_count"] == 648
    assert "read no holdout" in plan["holdout_rule"]
    assert plan["identity"]["seed"] == 20260720


def test_v28_training_decision_rejects_both_arms_without_strategy_holdout():
    decision_path = PLAN_PATH.with_name("v28_two_arm_training_decision.json")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))

    assert decision["status"] == "both_arms_rejected_after_training_gates"
    assert decision["strategy_holdout_evaluated"] is False
    assert decision["holdout_diagnostics_files_created"] is False
    assert decision["strict_factor_weight_winner"]["gate_results"][
        "passes_all_gates"
    ] is False
    assert decision["fold_boundary_issue"][
        "weight_winner_ga_fold_three_calmar"
    ] > decision["fold_boundary_issue"][
        "weight_winner_calendar_2016_2018_calmar"
    ]
