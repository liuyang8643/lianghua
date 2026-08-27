from core.ga import get_profile


def test_v38_profile_is_a_low_dimensional_v31_extension():
    profile = get_profile("v38_smallcap_trend_consistency")

    assert [cls.__name__ for cls in profile["factor_classes"]] == [
        "PreCloseMarketCap",
        "AmihudIlliquidityStrict",
        "TrendReversalPreCloseStrict",
        "VolumeCVStrict",
        "CompletedSmallCapTrendConsistency60Strict",
    ]
    assert profile["fixed_weights"] == {
        "PreCloseMarketCap": 1.0,
        "AmihudIlliquidityStrict": 0.4,
        "VolumeCVStrict": 0.2,
    }
    assert profile["search_spaces"]["buy_n"] == [30]
    assert profile["constraints"]["sell_m_equals_buy_n"] is True

    fixed = profile["fixed_parameters"]
    assert fixed["stock_pool"] == ["60", "00", "30"]
    assert fixed["trend_risk_overlay"]["mode"] == "dual_completed"
    assert fixed["trend_risk_overlay"]["strategy_weight"] == 0.8

    objective = profile["training_objective"]
    assert objective["mode"] == "robust_calmar"
    assert objective["calendar_folds"] == [
        ["2010-01-01", "2012-12-31"],
        ["2013-01-01", "2015-12-31"],
        ["2016-01-01", "2018-12-31"],
    ]
