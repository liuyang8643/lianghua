from core.ga import get_profile


def test_v40_profile_is_one_fixed_low_beta_extension_of_v31():
    profile = get_profile("v40_low_beta_fixed")

    assert [cls.__name__ for cls in profile["factor_classes"]] == [
        "PreCloseMarketCap",
        "AmihudIlliquidityStrict",
        "TrendReversalPreCloseStrict",
        "VolumeCVStrict",
        "Completed6030PoolBeta60Strict",
    ]
    assert profile["fixed_weights"] == {
        "PreCloseMarketCap": 1.0,
        "AmihudIlliquidityStrict": 0.4,
        "TrendReversalPreCloseStrict": 0.1,
        "VolumeCVStrict": 0.2,
    }
    assert profile["search_spaces"]["buy_n"] == [30]
    assert profile["constraints"]["sell_m_equals_buy_n"] is True

    fixed = profile["fixed_parameters"]
    assert fixed["stock_pool"] == ["60", "00", "30"]
    assert fixed["holding_period"] == 1
    assert fixed["trend_risk_overlay"]["mode"] == "dual_completed"
    assert fixed["trend_risk_overlay"]["strategy_weight"] == 0.8

    objective = profile["training_objective"]
    assert objective["mode"] == "robust_calmar"
    assert objective["calendar_folds"] == [
        ["2010-01-01", "2012-12-31"],
        ["2013-01-01", "2015-12-31"],
        ["2016-01-01", "2018-12-31"],
    ]
