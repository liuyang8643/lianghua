from core.ga._profiles import get_profile


def test_v24_adds_only_strict_downside_impact_to_v22_factors():
    profile = get_profile("v24_preclose_official_strict_downside_impact")
    names = [factor.__name__ for factor in profile["factor_classes"]]

    assert names == [
        "PreCloseMarketCap",
        "AmihudIlliquidityStrict",
        "TrendReversalPreCloseStrict",
        "VolumeCVStrict",
        "DownsideAmihudIlliquidityStrict",
    ]
    assert all("Gap" not in name for name in names)
    assert profile["fixed_weights"] == {"PreCloseMarketCap": 1.0}
    assert profile["search_spaces"] == {"buy_n": [40]}
    assert profile["fixed_parameters"]["stock_pool"] == ["60", "00", "30"]
