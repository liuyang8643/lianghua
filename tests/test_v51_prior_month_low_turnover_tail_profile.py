from core.ga import build_individual_config, get_profile


def test_v51_profile_is_one_fixed_prior_month_tail_interaction():
    profile = get_profile("v51_prior_month_low_turnover_tail")

    assert [factor.__name__ for factor in profile["factor_classes"]] == [
        "PreCloseMarketCap",
        "AmihudIlliquidityStrict",
        "TrendReversalPreCloseStrict",
        "VolumeCVStrict",
        "CompletedPriorMonthTurnoverStrict",
    ]
    assert profile["weight_search_spaces"] is None
    assert profile["fixed_weights"] == {
        "PreCloseMarketCap": 1.0,
        "AmihudIlliquidityStrict": 0.4,
        "TrendReversalPreCloseStrict": 0.1,
        "VolumeCVStrict": 0.2,
        "CompletedPriorMonthTurnoverStrict": 0.0,
    }
    assert profile["search_spaces"] == {"buy_n": [30]}
    assert profile["fixed_parameters"]["selection_sleeves"] == [
        {
            "name": "v31_core",
            "slots": 24,
            "weights": {
                "PreCloseMarketCap": 1.0,
                "AmihudIlliquidityStrict": 0.4,
                "TrendReversalPreCloseStrict": 0.1,
                "VolumeCVStrict": 0.2,
            },
        },
        {
            "name": "v31_tail_with_prior_month_low_turnover",
            "slots": 6,
            "weights": {
                "PreCloseMarketCap": 1.0,
                "AmihudIlliquidityStrict": 0.4,
                "TrendReversalPreCloseStrict": 0.1,
                "VolumeCVStrict": 0.2,
                "CompletedPriorMonthTurnoverStrict": 0.1,
            },
        },
    ]

    config = build_individual_config(
        buy_n=30,
        profile_name="v51_prior_month_low_turnover_tail",
    )
    assert config["selection_sleeves"] == profile["fixed_parameters"][
        "selection_sleeves"
    ]
    assert config["sell_m"] == 30
