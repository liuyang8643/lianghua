import json
from pathlib import Path

import numpy as np
import pytest

from env.action_schema import CORE_FACTOR_NAMES, CORE_FILTER_NAMES, ActionSchema
from env.contracts import DayConfig


ROOT = Path(__file__).resolve().parents[1]


def _current_static_config() -> dict:
    return json.loads((ROOT / "configs" / "config.json").read_text(encoding="utf-8"))


def test_action_layout_and_box_bounds_are_stable():
    schema = ActionSchema()

    assert schema.schema_version == "day-config-v20-turnover-floor"
    assert (schema.layout[-1].minimum, schema.layout[-1].maximum) == (0.05, 0.2)
    assert schema.action_dim == 12
    assert schema.action_names == (
        *(f"factor_weight.{name}" for name in CORE_FACTOR_NAMES),
        "turnover_rate",
    )
    low, high = schema.space_bounds
    np.testing.assert_array_equal(low, np.full(12, -1.0, dtype=np.float32))
    np.testing.assert_array_equal(high, np.full(12, 1.0, dtype=np.float32))
    assert low.dtype == high.dtype == np.float32


def test_decode_handles_boundaries_with_fixed_controls():
    schema = ActionSchema()

    low_action = np.full(schema.action_dim, -1.0, dtype=np.float32)
    assert not any(schema.decode(low_action).factor_enabled.values())
    low_action[0] = -0.5
    low = schema.decode(low_action)
    assert low.buy_n == 50
    assert low.turnover_rate == 0.05  # production floor: always examine the two worst holdings
    assert low.replacement_limit == 2
    assert low.limit_up_protection is True
    assert low.rebalance_band_pct == 0.01
    assert low.single_buy_pct == pytest.approx(1.0 / low.buy_n)
    assert tuple(low.factor_enabled.values()) == (True,) + (False,) * 10
    assert tuple(low.factor_weights.values()) == pytest.approx((0.25,) + (0.0,) * 10)

    high = schema.decode(np.full(schema.action_dim, 1.0, dtype=np.float32))
    assert high.buy_n == 50
    assert high.turnover_rate == pytest.approx(0.2)
    assert high.replacement_limit == 10
    assert high.limit_up_protection is True
    assert high.rebalance_band_pct == 0.01
    assert high.single_buy_pct == pytest.approx(1.0 / high.buy_n)
    assert all(high.factor_enabled.values())
    assert tuple(high.factor_weights.values()) == pytest.approx((1.0,) * 11)


def test_turnover_is_continuous_and_roundtrips_without_a_codebook():
    schema = ActionSchema()
    rates = []
    for coordinate in np.linspace(-1, 1, 301, dtype=np.float32):
        action = np.zeros(schema.action_dim, dtype=np.float32)
        action[-1] = coordinate
        decoded = schema.decode(action)
        assert decoded.buy_n == 50 and decoded.single_buy_pct == .02
        assert decoded.turnover_rate == pytest.approx(0.05 + (float(coordinate) + 1.0) / 2.0 * 0.15, abs=6e-8)
        np.testing.assert_array_equal(schema.encode(schema.canonicalize_day_config(decoded)), schema.encode(decoded))
        rates.append(decoded.turnover_rate)
    assert len(set(rates)) == 301
    assert all(field.kind == "continuous" for field in schema.layout)
    assert "turnover_rate_choices" not in schema.to_dict()


def test_turnover_rate_is_independent_of_fixed_portfolio_size():
    small, large = ActionSchema(), ActionSchema(fixed_buy_n=300)
    action = np.zeros(small.action_dim, dtype=np.float32)
    action[-1] = 0.0  # unit 0.5 -> 0.05 + 0.5 * 0.15 = 12.5%, independent of buy_n
    left, right = small.decode(action), large.decode(action)
    assert left.turnover_rate == right.turnover_rate == pytest.approx(0.125)
    assert left.replacement_limit == 6
    assert right.replacement_limit == 37
    assert small.action_dim == large.action_dim == 12


def test_schema_payload_rejects_removed_discrete_metadata():
    schema = ActionSchema()
    assert ActionSchema.from_dict(schema.to_dict()) == schema
    payload = {**schema.to_dict(), "turnover_rate_choices": [0.0, 1.0]}
    with pytest.raises(ValueError, match="payload mismatch"):
        ActionSchema.from_dict(payload)


def test_decode_is_deterministic_and_rejects_invalid_vectors():
    schema = ActionSchema()
    action = np.linspace(-1.0, 1.0, schema.action_dim, dtype=np.float32)

    assert schema.decode(action) == schema.decode(action.copy())
    with pytest.raises(ValueError, match="shape"):
        schema.decode(action[:-1])
    action[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        schema.decode(action)


def test_current_config_roundtrips_without_any_exposure_control():
    schema = ActionSchema()
    payload = _current_static_config()
    expected = schema.from_static_config(payload)

    assert expected.factor_weights == pytest.approx(
        {
            "TrueMarketCap": 0.5625,
            "VolumeCV": 0.0625,
            "AmountBasedSmallCap": 0.375,
            "CompletedReversal20": 0.0,
            "CompletedMomentum252Skip21": 0.0,
            "BiliAdjustedIssueDiscount": 0.0,
            "BiliHighLifetimeRangeRatio": 0.0,
            "PBBelowTwoROEAbove10Signal": 0.0,
            "LowCashOutflowProfitGrowthSpread": 0.0,
            "HighOperatingProfitRevenueGrowthSpread": 0.0,
            "HighAbnormalGrossProfit": 0.0,
        }
    )
    assert tuple(expected.factor_enabled.values()) == (True,) * 3 + (False,) * 8
    assert all(expected.filter_flags.values())
    assert expected.buy_n == 50
    assert expected.turnover_rate == pytest.approx(0.1, abs=6e-8)
    assert expected.replacement_limit == 5
    assert schema.canonicalize_day_config(expected) == expected
    assert expected.limit_up_protection is True
    assert expected.rebalance_band_pct == pytest.approx(0.01)
    assert expected.single_buy_pct == pytest.approx(1.0 / expected.buy_n)

    action = schema.encode(expected)
    decoded = schema.decode(action)
    assert decoded.factor_weights == pytest.approx(expected.factor_weights, abs=1e-7)
    assert decoded.factor_enabled == expected.factor_enabled
    assert decoded.filter_flags == expected.filter_flags
    assert decoded.buy_n == expected.buy_n
    assert decoded.turnover_rate == pytest.approx(expected.turnover_rate, abs=1e-7)
    assert decoded.limit_up_protection == expected.limit_up_protection
    assert decoded.rebalance_band_pct == pytest.approx(expected.rebalance_band_pct, abs=1e-7)
    assert decoded.single_buy_pct == pytest.approx(expected.single_buy_pct, abs=1e-7)

    exported = schema.to_static_config(decoded)
    for removed in (
        "position_multiplier",
        "target_exposure",
        "cash_reserve_ratio",
        "rebalance_now",
        "rebalance",
        "holding_period",
        "timing_enabled",
        "trend_risk_overlay",
        "exposure_step",
        "prefilter_n",
    ):
        assert removed not in exported
    assert exported["weights"] == pytest.approx(expected.factor_weights, abs=1e-7)


def test_static_config_rejects_unregistered_filter_instead_of_ignoring_it():
    schema = ActionSchema()
    legacy = _current_static_config()
    legacy["individual_config"]["filter_factors"]["UnknownFilter"] = True

    with pytest.raises(ValueError, match="filter vocabulary"):
        schema.from_static_config(legacy)


def test_serialized_day_config_restore_is_exact_and_has_no_defaults():
    schema = ActionSchema()
    action = np.zeros(schema.action_dim, dtype=np.float32)
    action[0] = -0.5
    config = schema.decode(action)
    serialized = schema.to_static_config(config)

    assert schema.from_serialized_day_config(serialized) == config

    missing_outer = dict(serialized)
    missing_outer.pop("turnover_rate")
    with pytest.raises(ValueError, match="turnover_rate"):
        schema.from_serialized_day_config(missing_outer)

    unexpected_outer = {**serialized, "legacy_default": True}
    with pytest.raises(ValueError, match="legacy_default"):
        schema.from_serialized_day_config(unexpected_outer)

    missing_factor = dict(serialized)
    missing_factor["factor_enabled"] = dict(serialized["factor_enabled"])
    missing_factor["factor_enabled"].pop(CORE_FACTOR_NAMES[0])
    with pytest.raises(ValueError, match="factor_enabled vocabulary"):
        schema.from_serialized_day_config(missing_factor)

    unexpected_filter = dict(serialized)
    unexpected_filter["filter_factors"] = {
        **serialized["filter_factors"],
        "unregistered_filter": False,
    }
    with pytest.raises(ValueError, match="filter vocabulary"):
        schema.from_serialized_day_config(unexpected_filter)


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("cash_reserve_ratio", 0.0),
        ("cash_reserve", 0.0),
        ("position_multiplier", 1.0),
        ("position_multipliers", [1.0]),
        ("target_exposure", 1.0),
        ("target_cash", 0.0),
        ("target_positions", {}),
        ("rebalance_now", True),
        ("rebalance", True),
        ("rebalance_mode", "equalize"),
        ("rebalance_frequency", "daily"),
        ("holding_period", 1),
        ("timing_enabled", False),
        ("timing_base", None),
        ("timing_leverage", 0.0),
        ("empty_months", []),
        ("trend_risk_overlay", {"enabled": False}),
        ("trend_risk_gate", False),
        ("exposure_step", 0.0),
    ),
)
def test_static_config_rejects_removed_exposure_and_timing_controls(name, value):
    payload = _current_static_config()
    payload["individual_config"][name] = value

    with pytest.raises(ValueError, match="unexpected fields"):
        ActionSchema().from_static_config(payload)


def test_static_config_allowlist_rejects_unknown_inner_and_outer_fields():
    schema = ActionSchema()
    inner = _current_static_config()
    inner["individual_config"]["ignored_legacy_field"] = None
    with pytest.raises(ValueError, match="unexpected fields.*ignored_legacy_field"):
        schema.from_static_config(inner)

    outer = _current_static_config()
    outer["legacy_metadata"] = {}
    with pytest.raises(ValueError, match="wrapper.*legacy_metadata"):
        schema.from_static_config(outer)


def test_static_config_explicit_vocabularies_must_match_exactly():
    schema = ActionSchema()
    payload = _current_static_config()
    payload["individual_config"]["factor_enabled"] = dict.fromkeys(
        CORE_FACTOR_NAMES,
        True,
    )
    with pytest.raises(ValueError, match="unexpected fields.*factor_enabled"):
        schema.from_static_config(payload)

    payload = _current_static_config()
    payload["individual_config"]["filter_factors"].pop(CORE_FILTER_NAMES[0])
    with pytest.raises(ValueError, match="filter vocabulary"):
        schema.from_static_config(payload)


def test_day_config_accepts_unit_weights_and_rejects_enable_mismatch_and_hidden_cash():
    kwargs = {
        "factor_weights": dict.fromkeys(CORE_FACTOR_NAMES, 0.25),
        "factor_enabled": dict.fromkeys(CORE_FACTOR_NAMES, True),
        "filter_flags": dict.fromkeys(CORE_FILTER_NAMES, True),
        "buy_n": 50,
        "turnover_rate": .1,
        "limit_up_protection": True,
        "rebalance_band_pct": 0.01,
        "single_buy_pct": 0.05,
    }
    invalid_disabled = dict(kwargs)
    invalid_disabled["factor_enabled"] = {
        **kwargs["factor_enabled"],
        CORE_FACTOR_NAMES[0]: False,
    }
    with pytest.raises(ValueError, match="zero weight"):
        DayConfig(**invalid_disabled)

    unit = dict(kwargs)
    unit["factor_weights"] = {
        name: (0.8 if index % 2 else 0.6)
        for index, name in enumerate(CORE_FACTOR_NAMES)
    }
    DayConfig(**unit)

    outside = dict(kwargs)
    outside["factor_weights"] = {**kwargs["factor_weights"], CORE_FACTOR_NAMES[0]: -0.01}
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        DayConfig(**outside)

    hidden_cash = dict(kwargs, single_buy_pct=0.019)
    with pytest.raises(ValueError, match="1 / buy_n"):
        DayConfig(**hidden_cash)
