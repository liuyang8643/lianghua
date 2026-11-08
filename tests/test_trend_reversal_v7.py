import numpy as np

from factor_db.factors.TrendReversalV7 import TrendReversalV7


def _panel(days=90, stocks=3):
    t = np.arange(days, dtype=np.float64)[:, None]
    s = np.arange(stocks, dtype=np.float64)[None, :]
    close = 10.0 + 0.05 * t + s
    high = close + 0.4
    low = close - 0.4
    return {
        "open": close.copy(),
        "high": high,
        "low": low,
        "close": close,
        "st_mask": np.zeros((days, stocks), dtype=bool),
    }


def test_v7_is_legal_and_dense_after_history():
    result = TrendReversalV7().calc_batch(_panel())
    assert np.all(result[:61] == 0.5)
    assert np.isfinite(result[61:]).all()
    assert np.all((result >= 0.0) & (result <= 1.0))


def test_v7_current_non_open_fields_cannot_change_score():
    panel = _panel()
    expected = TrendReversalV7().calc_batch(panel)[-1]
    changed = {key: value.copy() for key, value in panel.items()}
    for key in ("high", "low", "close"):
        changed[key][-1] = np.array([0.1, 1000.0, 0.2])
    np.testing.assert_array_equal(TrendReversalV7().calc_batch(changed)[-1], expected)
