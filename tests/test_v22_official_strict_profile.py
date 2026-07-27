from core.ga._profiles import get_profile


def test_v22_profile_contains_only_strict_non_gap_factors():
    profile = get_profile("v22_preclose_official_strict_control")
    names = [factor.__name__ for factor in profile["factor_classes"]]

    assert names == [
        "PreCloseMarketCap",
        "AmihudIlliquidityStrict",
        "TrendReversalPreCloseStrict",
        "VolumeCVStrict",
    ]
    assert all("Gap" not in name for name in names)
    assert profile["fixed_weights"] == {"PreCloseMarketCap": 1.0}
    assert profile["search_spaces"] == {"buy_n": [40]}
    assert profile["fixed_parameters"]["stock_pool"] == ["60", "00", "30"]
    assert profile["constraints"] == {"sell_m_equals_buy_n": True}
