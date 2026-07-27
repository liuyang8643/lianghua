from core.ga._profiles import get_profile


def test_v23_adds_only_fixed_gross_margin_to_v22_factors():
    profile = get_profile("v23_preclose_official_strict_gross_margin")
    names = [factor.__name__ for factor in profile["factor_classes"]]

    assert names == [
        "PreCloseMarketCap",
        "AmihudIlliquidityStrict",
        "TrendReversalPreCloseStrict",
        "VolumeCVStrict",
        "GrossMarginQuality",
    ]
    assert all("Gap" not in name for name in names)
    assert profile["fixed_weights"] == {"PreCloseMarketCap": 1.0}
    assert profile["search_spaces"] == {"buy_n": [40]}
    assert profile["fixed_parameters"]["stock_pool"] == ["60", "00", "30"]
