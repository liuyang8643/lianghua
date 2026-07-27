import numpy as np
import pytest

from factor_db.factors.AmihudIlliquidity import AmihudIlliquidity
from factor_db.factors.AmountBasedSmallCap import AmountBasedSmallCap
from factor_db.factors.IntermediateMomentum12Minus1 import (
    IntermediateMomentum12Minus1,
)
from factor_db.factors.TrendReversalV7 import TrendReversalV7
from factor_db.factors.TrueMarketCap import TrueMarketCap
from factor_db.factors.VolumeCV import VolumeCV


FACTOR_CLASSES = (
    AmihudIlliquidity,
    TrueMarketCap,
    VolumeCV,
    AmountBasedSmallCap,
    TrendReversalV7,
    IntermediateMomentum12Minus1,
)


def _panel(days=300, stocks=3):
    day = np.arange(days, dtype=np.float64)[:, None]
    stock = np.arange(stocks, dtype=np.float64)[None, :]
    close = 10.0 + 0.015 * day + 0.2 * stock + 0.1 * np.sin(day / 7.0 + stock)
    pre_close = np.empty_like(close)
    pre_close[0] = close[0]
    pre_close[1:] = close[:-1]
    volume = 1_000_000.0 + 10_000.0 * day + 50_000.0 * stock
    return {
        "open": close * 1.001,
        "high": close * 1.02,
        "low": close * 0.98,
        "close": close,
        "preClose": pre_close,
        "volume": volume,
        "amount": volume * close,
        "st_mask": np.zeros(close.shape, dtype=bool),
        "total_share": np.full(close.shape, 500_000_000.0),
    }


@pytest.mark.parametrize("factor_class", FACTOR_CLASSES)
def test_v9_v10_factor_ignores_current_day_hlcv_and_amount(factor_class):
    panel = _panel()
    expected = factor_class().calc_batch(panel)

    changed = {name: values.copy() for name, values in panel.items()}
    for name in ("high", "low", "close", "volume", "amount"):
        changed[name][-1] *= 100.0
    actual = factor_class().calc_batch(changed)

    np.testing.assert_equal(actual[-1], expected[-1])


@pytest.mark.parametrize("factor_class", FACTOR_CLASSES)
def test_v9_v10_factor_future_rows_do_not_change_history(factor_class):
    panel = _panel()
    expected = factor_class().calc_batch(panel)
    cutoff = 260

    changed = {name: values.copy() for name, values in panel.items()}
    for name in ("open", "high", "low", "close", "preClose", "volume", "amount"):
        changed[name][cutoff + 1:] *= 100.0
    changed["st_mask"][cutoff + 1:] = True
    changed["total_share"][cutoff + 1:] *= 100.0
    actual = factor_class().calc_batch(changed)

    np.testing.assert_equal(actual[:cutoff + 1], expected[:cutoff + 1])
