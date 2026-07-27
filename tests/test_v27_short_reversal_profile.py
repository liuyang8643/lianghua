import json
from pathlib import Path

from core.ga._profiles import get_profile


ROOT = Path(__file__).resolve().parents[1]


def test_v27_profile_is_fixed_no_gap_single_variable_ablation():
    profile = get_profile("v27_preclose_official_short_reversal20")
    names = [factor.__name__ for factor in profile["factor_classes"]]

    assert names == [
        "PreCloseMarketCap",
        "AmihudIlliquidityStrict",
        "TrendReversalPreCloseStrict",
        "VolumeCVStrict",
        "ShortTermReversal20Strict",
    ]
    assert profile["fixed_weights"] == {
        "PreCloseMarketCap": 1.0,
        "AmihudIlliquidityStrict": 0.2,
        "TrendReversalPreCloseStrict": 0.1,
        "VolumeCVStrict": 0.1,
        "ShortTermReversal20Strict": 0.1,
    }
    timing = profile["fixed_parameters"]["trend_risk_overlay"]
    assert timing["mode"] == "dual_completed"
    assert timing["strict_history"] is True
    assert all("Gap" not in name for name in names)


def test_v27_config_differs_from_v25_only_by_preregistered_factor():
    results = ROOT / "results" / "strategy_opt_20260721"
    v25 = json.loads(
        (results / "v25_completed_timing_control_config.json").read_text(encoding="utf-8")
    )
    v27 = json.loads(
        (results / "v27_short_reversal_w10_config.json").read_text(encoding="utf-8")
    )

    assert v27["ga_profile"] == "v27_preclose_official_short_reversal20"
    v27_individual = v27["individual_config"]
    v25_individual = v25["individual_config"]
    assert v27_individual["weights"] == {
        **v25_individual["weights"],
        "ShortTermReversal20Strict": 0.1,
    }
    for key in v25_individual:
        if key != "weights":
            assert v27_individual[key] == v25_individual[key]


def test_v27_plan_was_preregistered_before_backtest():
    plan = json.loads(
        (
            ROOT
            / "results/strategy_opt_20260721/v27_short_reversal_preregistered_plan.json"
        ).read_text(encoding="utf-8")
    )
    assert plan["status"] == "preregistered_before_implementation_or_backtest"
    assert plan["single_variable"] == {
        "factor": "ShortTermReversal20Strict",
        "weight": 0.1,
        "formula": "-expm1(sum(log(close[d]/official_preClose[d]))) for exactly d in [T-20,T)",
        "missing_values": "Any non-finite or non-positive close/preClose observation invalidates the complete window; no fill or fallback is allowed.",
    }


def test_v27_training_decision_rejects_without_holdout():
    decision = json.loads(
        (
            ROOT
            / "results/strategy_opt_20260721/v27_short_reversal_training_decision.json"
        ).read_text(encoding="utf-8")
    )

    assert decision["status"] == "rejected_after_training_gates"
    assert decision["holdout_evaluated"] is False
    assert decision["gate_results"]["passes_all_gates"] is False
    assert decision["gate_results"]["trade_open_audit"] is True
    assert decision["gate_results"]["robust_train_score_exceeds_v25"] is False
