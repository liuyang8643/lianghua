from core.ga._profiles import get_profile


def test_v21_profile_is_low_dimensional_and_contains_no_gap_signal():
    profile = get_profile("v21_preclose_strict_volume_low_dim")
    names = [factor.__name__ for factor in profile["factor_classes"]]

    assert names == [
        "AmihudIlliquidity",
        "PreCloseMarketCap",
        "VolumeCVStrict",
        "AmountBasedSmallCap",
        "TrendReversalV7",
    ]
    assert all("Gap" not in name for name in names)
    assert profile["fixed_weights"] == {"PreCloseMarketCap": 1.0}
    assert profile["search_spaces"] == {"buy_n": [20, 30, 40, 50]}
    assert profile["fixed_parameters"]["stock_pool"] == ["60", "00", "30"]
    assert profile["constraints"] == {"sell_m_equals_buy_n": True}
    assert profile["training_objective"] == {
        "mode": "robust_calmar",
        "folds": 3,
        "full_weight": 0.5,
        "min_average_exposure": 0.45,
    }
