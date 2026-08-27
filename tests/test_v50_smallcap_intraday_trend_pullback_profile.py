from core.ga import build_individual_config, get_profile


def test_v50_profile_is_one_fixed_low_weight_tail_interaction():
    profile = get_profile("v50_smallcap_intraday_trend_pullback_tail")

    assert [factor.__name__ for factor in profile["factor_classes"]] == [
        "PreCloseMarketCap",
        "AmihudIlliquidityStrict",
        "TrendReversalPreCloseStrict",
        "VolumeCVStrict",
        "CompletedSmallCapIntradayTrendPullback60x5Strict",
    ]
    assert profile["weight_search_spaces"] is None
    assert profile["fixed_weights"] == {
        "PreCloseMarketCap": 1.0,
        "AmihudIlliquidityStrict": 0.4,
        "TrendReversalPreCloseStrict": 0.1,
        "VolumeCVStrict": 0.2,
        "CompletedSmallCapIntradayTrendPullback60x5Strict": 0.0,
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
            "name": "v31_tail_with_intraday_trend_pullback",
            "slots": 6,
            "weights": {
                "PreCloseMarketCap": 1.0,
                "AmihudIlliquidityStrict": 0.4,
                "TrendReversalPreCloseStrict": 0.1,
                "VolumeCVStrict": 0.2,
                "CompletedSmallCapIntradayTrendPullback60x5Strict": 0.05,
            },
        },
    ]

    config = build_individual_config(
        buy_n=30,
        profile_name="v50_smallcap_intraday_trend_pullback_tail",
    )
    assert config["selection_sleeves"] == profile["fixed_parameters"][
        "selection_sleeves"
    ]
    assert config["sell_m"] == 30
