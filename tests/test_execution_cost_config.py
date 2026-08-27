import math

import numpy as np
import pytest

from core.fees import (
    SIM_SLIPPAGE_BPS,
    SIM_SLIPPAGE_RATE,
    simulated_buy_fee_rate,
    slippage_rate_from_bps,
)
from core.ga import get_profile, get_profile_search_spaces
from core.rebalance import compute_rebalance_plan
from core.strategy_config import normalize_individual_config


def _plan(**overrides):
    kwargs = {
        "positions": {},
        "sellable_volumes": {},
        "pos_vals": {},
        "cash": 0.0,
        "buy_n_stocks": [],
        "tradable_buy_stocks": [],
        "sellable_ok": set(),
        "prices": {},
        "limit_prices": {},
        "base_target": 0.0,
        "rebalance": True,
    }
    kwargs.update(overrides)
    return compute_rebalance_plan(**kwargs)


def test_default_slippage_is_the_existing_ten_bps_per_side():
    config = normalize_individual_config({
        "weights": {"Score": 1.0},
        "buy_n": 1,
    })

    assert config["slippage_bps"] == 10.0
    assert config["rebalance_band_pct"] == 0.01
    assert SIM_SLIPPAGE_BPS == 10.0
    assert slippage_rate_from_bps(config["slippage_bps"]) == SIM_SLIPPAGE_RATE


def test_profile_constraint_normalizes_sell_m_before_costs():
    legacy = {
        "weights": {"Score": 1.0},
        "buy_n": 10,
        "sell_m": 20,
    }

    normalized = normalize_individual_config(legacy, "v9_dual_shadow")

    assert normalized["sell_m"] == 10
    assert normalized["slippage_bps"] == 10.0
    assert normalized["rebalance_band_pct"] == 0.01


@pytest.mark.parametrize(
    "value",
    [True, -0.01, float("nan"), float("inf"), 10_000],
)
def test_invalid_slippage_fails_closed(value):
    with pytest.raises(ValueError, match="slippage_bps"):
        normalize_individual_config({
            "weights": {"Score": 1.0},
            "buy_n": 1,
            "slippage_bps": value,
        })


@pytest.mark.parametrize(
    "value",
    [True, -0.01, 1.0, float("nan"), float("inf")],
)
def test_invalid_rebalance_band_fails_closed(value):
    with pytest.raises(ValueError, match="rebalance_band_pct"):
        normalize_individual_config({
            "weights": {"Score": 1.0},
            "buy_n": 1,
            "rebalance_band_pct": value,
        })


def test_higher_slippage_is_used_by_rebalance_cash_budget():
    code = "600000.SH"
    common = {
        "cash": 11_020.0,
        "buy_n_stocks": [code],
        "tradable_buy_stocks": [code],
        "prices": {code: 10.0},
        "limit_prices": {code: 11.0},
        "base_target": 10_000.0,
    }

    _, low_cost_buys, _ = _plan(**common, slippage_bps=10.0)
    _, high_cost_buys, _ = _plan(**common, slippage_bps=50.0)

    assert low_cost_buys == {code: 1000}
    assert high_cost_buys == {code: 900}
    assert simulated_buy_fee_rate(50.0) > simulated_buy_fee_rate(10.0)


def test_rebalance_band_suppresses_only_small_target_weight_adjustments():
    code = "600000.SH"
    common = {
        "positions": {code: 1140},
        "sellable_volumes": {code: 1140},
        "pos_vals": {code: 11_400.0},
        "cash": 0.0,
        "buy_n_stocks": [code],
        "tradable_buy_stocks": [code],
        "sellable_ok": {code},
        "prices": {code: 10.0},
        "limit_prices": {code: 11.0},
        "base_target": 10_000.0,
    }

    narrow_sells, _, _ = _plan(**common, rebalance_band_pct=0.01)
    wide_sells, _, _ = _plan(**common, rebalance_band_pct=0.15)

    assert narrow_sells == [(code, 100)]
    assert wide_sells == []

    replacement_sells, _, _ = _plan(
        **{
            **common,
            "buy_n_stocks": [],
            "tradable_buy_stocks": [],
            "base_target": 0.0,
        },
        rebalance_band_pct=0.15,
    )
    assert replacement_sells == [(code, -1)]


def test_v52_profile_fixes_cost_and_only_searches_the_no_trade_band():
    profile = get_profile("v52_cost_aware_rebalance_band")

    assert profile["fixed_parameters"]["buy_n"] == 30
    assert profile["fixed_parameters"]["slippage_bps"] == 30.0
    assert get_profile_search_spaces("v52_cost_aware_rebalance_band") == {
        "rebalance_band_pct": [0.01, 0.03, 0.05, 0.08, 0.1, 0.15],
    }
    assert math.isclose(profile["training_objective"]["full_weight"], 0.5)
