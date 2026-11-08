import numpy as np
import pytest

from factor_db.factors.MarketCapCeilingFilter import (
    FilterMarketCapMax25Yi,
    FilterMarketCapMax28Yi,
    FilterMarketCapMax30Yi,
    FilterMarketCapMax32Yi,
    FilterMarketCapMax35Yi,
)


@pytest.mark.parametrize(
    ("factor_class", "limit"),
    [
        (FilterMarketCapMax25Yi, 25.0),
        (FilterMarketCapMax28Yi, 28.0),
        (FilterMarketCapMax30Yi, 30.0),
        (FilterMarketCapMax32Yi, 32.0),
        (FilterMarketCapMax35Yi, 35.0),
    ],
)
def test_market_cap_ceiling_uses_runtime_share_units(factor_class, limit):
    panel = {
        "open": np.array([[10.0, 10.0, 10.0]]),
        "total_share": np.array([[
            limit * 1e4 / 10.0,
            (limit + 0.01) * 1e4 / 10.0,
            (limit - 0.01) * 1e4 / 10.0,
        ]]),
    }

    actual = factor_class().calc_batch(panel)

    np.testing.assert_array_equal(np.isfinite(actual), [[True, False, True]])


def test_market_cap_ceiling_exposes_invalid_inputs():
    panel = {
        "open": np.array([[np.nan, 0.0, 10.0]]),
        "total_share": np.array([[1000.0, 1000.0, np.nan]]),
    }

    actual = FilterMarketCapMax30Yi().calc_batch(panel)

    assert np.all(np.isnan(actual))


def test_market_cap_ceiling_can_use_current_open():
    panel = {
        "open": np.array([[9.0], [11.0]]),
        "total_share": np.array([[30000.0], [30000.0]]),
    }

    actual = FilterMarketCapMax30Yi().calc_batch(panel)

    assert np.isfinite(actual[0, 0])
    assert np.isnan(actual[1, 0])
