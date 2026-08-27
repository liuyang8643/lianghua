import json
from pathlib import Path

import numpy as np
import pytest

from env.action_schema import CORE_FACTOR_NAMES, CORE_FILTER_NAMES, ActionSchema
from env.contracts import DayConfig, RebalanceMode


ROOT = Path(__file__).resolve().parents[1]


def _current_static_config() -> dict:
    return json.loads((ROOT / "configs" / "config.json").read_text(encoding="utf-8"))


def test_action_layout_and_box_bounds_are_stable():
    schema = ActionSchema()

    assert schema.action_dim == 18
    assert schema.action_names == (
        *(f"factor_weight.{name}" for name in CORE_FACTOR_NAMES),
        *(f"factor_enabled.{name}" for name in CORE_FACTOR_NAMES),
        *(f"filter_flag.{name}" for name in CORE_FILTER_NAMES),
        "target_exposure",
        "buy_n",
        "sell_m",
        "rebalance_now",
        "rebalance_mode",
        "limit_up_protection",
        "rebalance_band_pct",
    )
    low, high = schema.space_bounds
    np.testing.assert_array_equal(low, np.full(18, -1.0, dtype=np.float32))
    np.testing.assert_array_equal(high, np.full(18, 1.0, dtype=np.float32))
    assert schema.action_space.shape == (18,)
    assert schema.action_space.dtype == np.float32


def test_decode_handles_all_boundaries_and_repairs_cross_field_constraints():
    schema = ActionSchema()

    low = schema.decode(np.full(schema.action_dim, -1.0, dtype=np.float32))
    assert low.target_exposure == 0.0
    assert low.buy_n == schema.buy_n_choices[0]
    assert low.sell_m == low.buy_n
    assert low.rebalance_now is False
    assert low.rebalance_mode is RebalanceMode.EQUALIZE
    assert low.limit_up_protection is False
    assert low.rebalance_band_pct == 0.0
    assert sum(low.factor_enabled.values()) == 1
    assert sum(low.factor_weights.values()) == pytest.approx(1.0)

    high = schema.decode(np.full(schema.action_dim, 1.0, dtype=np.float32))
    assert high.target_exposure == 1.0
    assert high.buy_n == schema.buy_n_choices[-1]
    assert high.sell_m == schema.sell_m_choices[-1]
    assert high.rebalance_now is True
    assert high.rebalance_mode is RebalanceMode.REPLACE_ONLY
    assert high.limit_up_protection is True
    assert high.rebalance_band_pct == 0.15
    assert all(high.factor_enabled.values())
    assert sum(high.factor_weights.values()) == pytest.approx(1.0)

    invalid_sell = schema.encode(schema.from_static_config(_current_static_config()))
    invalid_sell[schema.action_names.index("buy_n")] = 1.0
    invalid_sell[schema.action_names.index("sell_m")] = -1.0
    repaired = schema.decode(invalid_sell)
    assert repaired.sell_m >= repaired.buy_n


def test_decode_is_deterministic_and_rejects_invalid_vectors():
    schema = ActionSchema()
    action = np.linspace(-1.0, 1.0, schema.action_dim, dtype=np.float32)

    assert schema.decode(action) == schema.decode(action.copy())
    with pytest.raises(ValueError, match="shape"):
        schema.decode(action[:-1])
    action[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        schema.decode(action)


def test_current_config_roundtrips_to_semantically_equivalent_day_config():
    schema = ActionSchema()
    payload = _current_static_config()
    expected = schema.from_static_config(payload)

    assert expected.factor_weights == pytest.approx(
        {
            "AmihudIlliquidity": 0.20,
            "TrueMarketCap": 0.45,
            "VolumeCV": 0.05,
            "AmountBasedSmallCap": 0.30,
        }
    )
    assert all(expected.factor_enabled.values())
    assert all(expected.filter_flags.values())
    assert expected.target_exposure == pytest.approx(0.75)
    assert expected.buy_n == 20
    assert expected.sell_m == 25
    assert expected.rebalance_now is True
    assert expected.rebalance_mode is RebalanceMode.EQUALIZE
    assert expected.limit_up_protection is True
    assert expected.rebalance_band_pct == pytest.approx(0.01)

    action = schema.encode_static_config(payload)
    decoded = schema.decode(action)
    assert decoded.factor_weights == pytest.approx(expected.factor_weights, abs=1e-7)
    assert decoded.factor_enabled == expected.factor_enabled
    assert decoded.filter_flags == expected.filter_flags
    assert decoded.target_exposure == pytest.approx(expected.target_exposure, abs=1e-7)
    assert decoded.buy_n == expected.buy_n
    assert decoded.sell_m == expected.sell_m
    assert decoded.rebalance_now == expected.rebalance_now
    assert decoded.rebalance_mode == expected.rebalance_mode
    assert decoded.limit_up_protection == expected.limit_up_protection
    assert decoded.rebalance_band_pct == pytest.approx(expected.rebalance_band_pct, abs=1e-7)

    legacy = schema.to_static_config(decoded)
    assert "prefilter_n" not in legacy
    assert legacy["cash_reserve_ratio"] == pytest.approx(0.25, abs=1e-7)
    assert legacy["holding_period"] == 1
    assert legacy["rebalance"] is True
    assert legacy["weights"] == pytest.approx(expected.factor_weights, abs=1e-7)


def test_day_config_rejects_disabled_weight_and_non_normalized_weights():
    kwargs = {
        "factor_weights": dict.fromkeys(CORE_FACTOR_NAMES, 0.25),
        "factor_enabled": dict.fromkeys(CORE_FACTOR_NAMES, True),
        "filter_flags": dict.fromkeys(CORE_FILTER_NAMES, True),
        "target_exposure": 0.75,
        "buy_n": 20,
        "sell_m": 25,
        "rebalance_now": True,
        "rebalance_mode": RebalanceMode.EQUALIZE,
        "limit_up_protection": True,
        "rebalance_band_pct": 0.01,
    }
    invalid_disabled = dict(kwargs)
    invalid_disabled["factor_enabled"] = {
        **kwargs["factor_enabled"],
        CORE_FACTOR_NAMES[0]: False,
    }
    with pytest.raises(ValueError, match="zero weight"):
        DayConfig(**invalid_disabled)

    invalid_sum = dict(kwargs)
    invalid_sum["factor_weights"] = dict.fromkeys(CORE_FACTOR_NAMES, 0.20)
    with pytest.raises(ValueError, match="sum to 1"):
        DayConfig(**invalid_sum)
