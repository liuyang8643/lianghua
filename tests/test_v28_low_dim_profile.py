from core.ga._profiles import get_profile


def test_v28_search_space_is_fixed_low_dimensional_and_no_gap():
    profile = get_profile("v28_preclose_official_strict_low_dim")
    names = [factor.__name__ for factor in profile["factor_classes"]]

    assert names == [
        "PreCloseMarketCap",
        "AmihudIlliquidityStrict",
        "TrendReversalPreCloseStrict",
        "VolumeCVStrict",
    ]
    assert profile["fixed_weights"] == {"PreCloseMarketCap": 1.0}
    assert profile["search_spaces"] == {"buy_n": [30, 40, 50]}
    assert profile["weight_search_spaces"] == {
        "AmihudIlliquidityStrict": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "TrendReversalPreCloseStrict": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "VolumeCVStrict": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    }
    assert 3 * 6**3 == 648
    assert all("Gap" not in name for name in names)


def test_v28_freezes_completed_only_timing_and_train_contract():
    profile = get_profile("v28_preclose_official_strict_low_dim")
    fixed = profile["fixed_parameters"]
    timing = fixed["trend_risk_overlay"]

    assert fixed["stock_pool"] == ["60", "00", "30"]
    assert fixed["holding_period"] == 1
    assert timing["mode"] == "dual_completed"
    assert timing["strict_history"] is True
    assert timing["strategy_weight"] == 0.8
    assert profile["training_objective"] == {
        "mode": "robust_calmar",
        "folds": 3,
        "full_weight": 0.5,
        "min_average_exposure": 0.45,
    }
    assert profile["constraints"] == {"sell_m_equals_buy_n": True}
    assert str(profile["preload_start_date"]) == "2010-01-01"
    assert str(profile["preload_end_date"]) == "2018-12-31"


def test_v28_timing_arm_freezes_selection_and_uses_same_train_contract():
    profile = get_profile("v28_completed_timing_grid")
    names = [factor.__name__ for factor in profile["factor_classes"]]

    assert names == [
        "PreCloseMarketCap",
        "AmihudIlliquidityStrict",
        "TrendReversalPreCloseStrict",
        "VolumeCVStrict",
    ]
    assert profile["fixed_weights"] == {
        "PreCloseMarketCap": 1.0,
        "AmihudIlliquidityStrict": 0.2,
        "TrendReversalPreCloseStrict": 0.1,
        "VolumeCVStrict": 0.1,
    }
    assert profile["weight_search_spaces"] is None
    assert profile["search_spaces"] == {"buy_n": [40]}
    assert profile["training_objective"] == {
        "mode": "robust_calmar",
        "folds": 3,
        "full_weight": 0.5,
        "min_average_exposure": 0.45,
    }
