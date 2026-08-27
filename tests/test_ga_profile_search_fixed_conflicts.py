import yaml
import pytest

import core.ga._profiles as profiles
from core.ga import build_individual_config, get_profile


@pytest.mark.parametrize(
    ("search_key", "definition", "fixed_parameters", "fixed_path"),
    [
        (
            "buy_n",
            {
                "key": "buy_n",
                "config_key": "buy_n",
                "type": "int",
            },
            {"buy_n": 20},
            "fixed_parameters.buy_n",
        ),
        (
            "strategy_momentum_center",
            {
                "key": "strategy_momentum_center",
                "config_group": "trend_risk_overlay",
                "config_key": "strategy_momentum_center",
                "type": "float",
            },
            {
                "trend_risk_overlay": {
                    "strategy_momentum_center": -0.044,
                },
            },
            (
                "fixed_parameters.trend_risk_overlay."
                "strategy_momentum_center"
            ),
        ),
        (
            "trend_floor",
            {
                "key": "trend_floor",
                "config_group": "trend_risk_overlay",
                "config_key": "floor",
                "type": "float",
            },
            {"trend_risk_overlay": {"floor": 0.0}},
            "fixed_parameters.trend_risk_overlay.floor",
        ),
        (
            "strategy_ma_center",
            {
                "key": "strategy_ma_center",
                "config_group": "trend_risk_overlay",
                "config_key": "strategy_ma_center",
                "type": "float",
            },
            {"trend_risk_overlay": "invalid-whole-group-value"},
            "fixed_parameters.trend_risk_overlay",
        ),
    ],
)
def test_profile_load_rejects_search_dimension_overwritten_by_fixed_value(
    tmp_path,
    monkeypatch,
    search_key,
    definition,
    fixed_parameters,
    fixed_path,
):
    yaml_path = tmp_path / "strategy.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "strategy_parameters": [definition],
                "profiles": {
                    "conflicting_profile": {
                        "search_spaces": [search_key],
                        "fixed_parameters": fixed_parameters,
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(profiles, "_YAML_PATH", yaml_path)
    monkeypatch.setattr(profiles, "_loaded", False)

    with pytest.raises(
        ValueError,
        match=(
            rf"conflicting_profile.*{search_key}.*"
            rf"{fixed_path}"
        ),
    ):
        profiles._load()

    assert profiles._loaded is False


def test_nested_search_does_not_conflict_with_other_fixed_overlay_fields():
    profiles._validate_search_fixed_parameter_conflicts(
        "valid_timing_profile",
        {
            "search_spaces": [
                "strategy_momentum_center",
                "strategy_ma_center",
                "exposure_step",
            ],
            "fixed_parameters": {
                "trend_risk_overlay": {
                    "strategy_weight": 1.0,
                    "strategy_momentum_scale": 0.015,
                    "strategy_ma_scale": 0.009,
                },
            },
            "fixed_weights": {"PreCloseMarketCap": 1.0},
        },
        [
            {
                "key": key,
                "config_group": "trend_risk_overlay",
                "config_key": key,
                "type": "float",
            }
            for key in (
                "strategy_momentum_center",
                "strategy_ma_center",
                "exposure_step",
            )
        ],
    )


@pytest.mark.parametrize(
    (
        "strategy_momentum_center",
        "strategy_ma_center",
        "exposure_step",
    ),
    [
        (-0.065, 1.005, 0.0),
        (-0.044, 1.020, 0.25),
    ],
)
def test_v64_searched_timing_values_survive_fixed_parameter_merge(
    strategy_momentum_center,
    strategy_ma_center,
    exposure_step,
):
    profile_name = "v64_fixed_count_strategy_timing_20bp"
    profile = get_profile(profile_name)
    fixed_overlay = profile["fixed_parameters"]["trend_risk_overlay"]

    assert profile["search_spaces"] == {
        "strategy_momentum_center": [-0.065, -0.055, -0.044],
        "strategy_ma_center": [1.005, 1.014, 1.02],
        "exposure_step": [0.0, 0.1, 0.2, 0.25],
    }
    assert "strategy_momentum_center" not in fixed_overlay
    assert "strategy_ma_center" not in fixed_overlay
    assert "exposure_step" not in fixed_overlay

    config = build_individual_config(
        profile_name=profile_name,
        strategy_momentum_center=strategy_momentum_center,
        strategy_ma_center=strategy_ma_center,
        exposure_step=exposure_step,
    )

    assert config["trend_risk_overlay"]["strategy_momentum_center"] == (
        strategy_momentum_center
    )
    assert config["trend_risk_overlay"]["strategy_ma_center"] == (
        strategy_ma_center
    )
    assert config["trend_risk_overlay"]["exposure_step"] == exposure_step


def test_fixed_weights_can_coexist_with_searchable_factor_weights():
    profile = get_profile("v44_v31_core_reweight")

    assert profile["fixed_weights"] == {"PreCloseMarketCap": 1.0}
    assert set(profile["weight_search_spaces"]) == {
        "AmihudIlliquidityStrict",
        "TrendReversalPreCloseStrict",
        "VolumeCVStrict",
    }
