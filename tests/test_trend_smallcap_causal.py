from datetime import datetime, timedelta

import numpy as np

from factor_db.factors.TrendSmallCapCausal import TrendSmallCapCausal


def _panel(days=260, stocks=5):
    close = np.linspace(8.0, 12.0, days)[:, None] * np.linspace(0.95, 1.05, stocks)[None, :]
    open_ = close * 1.001
    amount = np.full(close.shape, 1e8)
    return {
        "open": open_,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "preClose": np.vstack((close[:1], close[:-1])),
        "amount": amount,
        "volume": amount / close,
        "st_mask": np.zeros(close.shape, dtype=bool),
        "total_share": np.full(close.shape, 1e8),
        "trade_dates": np.array(
            [(datetime(2020, 1, 1) + timedelta(days=i)).date() for i in range(days)],
            dtype="datetime64[D]",
        ),
    }


def test_trend_uses_only_completed_price_path_for_current_row():
    panel = _panel()
    expected = TrendSmallCapCausal().calc_batch(panel)
    changed = {key: value.copy() if hasattr(value, "copy") else value for key, value in panel.items()}
    for key in ("high", "low", "close", "amount", "volume", "preClose"):
        changed[key][-1] = np.linspace(0.01, 999.0, panel[key].shape[1])
    np.testing.assert_array_equal(TrendSmallCapCausal().calc_batch(changed)[-1], expected[-1])


def test_trend_open_is_only_a_legality_gate():
    panel = _panel()
    expected = TrendSmallCapCausal().calc_batch(panel)
    changed = {key: value.copy() if hasattr(value, "copy") else value for key, value in panel.items()}
    changed["open"][-1] *= np.linspace(0.8, 1.2, panel["open"].shape[1])
    np.testing.assert_array_equal(TrendSmallCapCausal().calc_batch(changed)[-1], expected[-1])
    changed["open"][-1, 0] = 1.99
    assert TrendSmallCapCausal().calc_batch(changed)[-1, 0] != expected[-1, 0]


def test_insufficient_trend_history_is_neutral_not_missing():
    panel = _panel(days=50)
    result = TrendSmallCapCausal().calc_batch(panel)
    assert np.allclose(result[40], 0.5)
