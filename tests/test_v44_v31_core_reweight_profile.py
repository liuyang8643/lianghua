from core.ga import get_profile


def test_v44_profile_searches_only_three_v31_core_weights():
    profile = get_profile("v44_v31_core_reweight")

    assert [cls.__name__ for cls in profile["factor_classes"]] == [
        "PreCloseMarketCap",
        "AmihudIlliquidityStrict",
        "TrendReversalPreCloseStrict",
        "VolumeCVStrict",
    ]
    assert profile["fixed_weights"] == {"PreCloseMarketCap": 1.0}
    assert profile["weight_search_spaces"] == {
        "AmihudIlliquidityStrict": [
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
        ],
        "TrendReversalPreCloseStrict": [
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
        ],
        "VolumeCVStrict": [
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
        ],
    }
    assert profile["search_spaces"] == {"buy_n": [30]}
    assert profile["constraints"]["sell_m_equals_buy_n"] is True

    fixed = profile["fixed_parameters"]
    assert fixed["stock_pool"] == ["60", "00", "30"]
    assert fixed["holding_period"] == 1
    assert fixed["rebalance"] is True
    assert fixed["limit_up_protection"] is True
    assert fixed["trend_risk_overlay"]["mode"] == "dual_completed"
    assert fixed["trend_risk_overlay"]["strategy_weight"] == 0.8

    assert profile["training_objective"] == {
        "mode": "robust_calmar",
        "full_weight": 0.5,
        "min_average_exposure": 0.45,
        "calendar_folds": [
            ["2010-01-01", "2012-12-31"],
            ["2013-01-01", "2015-12-31"],
            ["2016-01-01", "2018-12-31"],
        ],
    }
