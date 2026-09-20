"""Affordability must match the former monotone search at exact cash edges."""
import numpy as np
import pytest

from env.fees import FeeSchedule
from env.quantity import floor_buy_quantity


def _binary_oracle(fees, cash, price):
    low, high = 0, int(cash / price) + 1
    while low + 1 < high:
        mid = (low + high) // 2
        if fees.buy_total_cost(mid * price) <= cash:
            low = mid
        else:
            high = mid
    return low


@pytest.mark.parametrize('fees', [
    FeeSchedule(), FeeSchedule(minimum_commission=5),
    FeeSchedule(commission_rate=0, minimum_commission=0, transfer_fee_rate=0, slippage_rate=0),
    FeeSchedule(commission_rate=.01, minimum_commission=10, transfer_fee_rate=.002, slippage_rate=.1),
])
def test_affordability_exact_boundaries_and_random_cash(fees):
    rng = np.random.default_rng(91842)
    for _ in range(250):
        price = float(10 ** rng.uniform(-2, 4))
        shares = int(rng.integers(1, 10_000_000))
        boundary = fees.buy_total_cost(shares * price)
        for cash in (np.nextafter(boundary, -np.inf), boundary,
                     np.nextafter(boundary, np.inf), float(10 ** rng.uniform(-3, 14))):
            expected = _binary_oracle(fees, cash, price)
            actual = fees.affordable_buy_shares(cash, price)
            assert actual == expected
            for code in ('000001.SZ', '688001.SH', '430001.BJ'):
                assert floor_buy_quantity(code, actual) == floor_buy_quantity(code, expected)


def test_less_than_minimum_commission_or_nonpositive_budget():
    fees = FeeSchedule(minimum_commission=5)
    assert fees.affordable_buy_shares(4.99, .01) == 0
    assert fees.affordable_buy_shares(0, 10) == 0
    assert fees.affordable_buy_shares(-1, 10) == 0
    assert fees.affordable_buy_shares(100, 0) == 0
