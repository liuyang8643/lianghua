import numpy as np

from factor_db.factors.TrendReversalV11 import TrendReversalV11
from factor_db.factors.TrendReversalV7 import TrendReversalV7


def _panel(days=80, stocks=4):
    time = np.arange(days, dtype=np.float64)[:, None]
    stock = np.arange(stocks, dtype=np.float64)[None, :]
    close = 10.0 + 0.03 * time + stock
    return {
        "open": close + 0.01,
        "high": close + 0.40,
        "low": close - 0.40,
        "close": close,
        "volume": 1_000_000.0 + 100.0 * time + stock,
        "amount": 10_000_000.0 + 1_000.0 * time + stock,
        "st_mask": np.zeros((days, stocks), dtype=bool),
    }


def test_exposes_insufficient_history_and_non_finite_components_as_nan():
    panel = _panel()
    panel["close"][40, 1] = np.nan
    panel["high"][50, 2] = np.inf
    result = TrendReversalV11().calc_batch(panel)

    assert np.isnan(result[:61]).all()
    assert np.isfinite(result[61, 0])
    assert np.isnan(result[61, 1])
    assert np.isnan(result[61, 2])
    assert np.isfinite(result[61, 3])


def test_matches_v7_formula_when_all_components_are_finite():
    panel = _panel()
    strict = TrendReversalV11().calc_batch(panel)
    v7 = TrendReversalV7().calc_batch(panel)

    np.testing.assert_array_equal(strict[61:], v7[61:])


def test_current_day_hlcv_and_amount_cannot_change_score():
    panel = _panel()
    factor = TrendReversalV11()
    expected = factor.calc_batch(panel)[-1]
    changed = {name: values.copy() for name, values in panel.items()}
    for name in ("high", "low", "close", "volume", "amount"):
        changed[name][-1] = np.array([np.nan, np.inf, -1.0, 1e30])

    actual = factor.calc_batch(changed)[-1]
    np.testing.assert_array_equal(actual, expected)


def test_current_open_and_st_mask_are_legality_gates_only():
    panel = _panel()
    factor = TrendReversalV11()
    expected = factor.calc_batch(panel)[-1]
    assert np.isfinite(expected).all()

    panel["open"][-1] = np.array([2.0, 2000.0, 1.99, np.nan])
    panel["st_mask"][-1, 1] = True
    actual = factor.calc_batch(panel)[-1]

    np.testing.assert_allclose(actual[0], expected[0])
    assert np.isnan(actual[1:]).all()
