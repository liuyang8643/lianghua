import numpy as np
import pytest

from research_train_cost_stress import stress_daily_returns


def test_cost_stress_uses_two_sides_of_reported_one_way_turnover():
    daily = np.array([1.0, -2.0, 3.0])
    rebalance_pct = np.array([0.0, 25.0, 100.0])

    actual = stress_daily_returns(daily, rebalance_pct, 5.0)

    np.testing.assert_allclose(actual, [1.0, -2.025, 2.9])


@pytest.mark.parametrize(
    ("daily", "rebalance", "message"),
    [
        (np.zeros((2, 1)), np.zeros(2), "one-dimensional"),
        (np.zeros(2), np.zeros(3), "matching lengths"),
        (np.array([np.nan]), np.zeros(1), "must be finite"),
        (np.zeros(1), np.array([-1.0]), "cannot be negative"),
    ],
)
def test_cost_stress_rejects_invalid_inputs(daily, rebalance, message):
    with pytest.raises(ValueError, match=message):
        stress_daily_returns(daily, rebalance, 5.0)


@pytest.mark.parametrize("bps", [-1.0, np.nan, np.inf])
def test_cost_stress_rejects_invalid_scenario(bps):
    with pytest.raises(ValueError, match="basis points"):
        stress_daily_returns(np.zeros(1), np.zeros(1), bps)
