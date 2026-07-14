import numpy as np
from datetime import datetime, timedelta

from factor_db.factors.TrendOpenSignalV2 import TrendOpenSignalV2


def _panel(days=100, stocks=25):
    base = np.linspace(8.0, 10.0, days)[:, None] * np.ones((1, stocks))
    amount = np.full_like(base, 1e8)
    start = datetime(2026, 1, 1)
    return {
        "open": base.copy(), "high": base * 1.01, "low": base * 0.99,
        "close": base.copy(), "amount": amount,
        "st_mask": np.zeros_like(base, dtype=bool),
        "preClose": np.vstack((base[:1], base[:-1])),
        "issue_price": np.full(stocks, 8.0),
        "stock_codes": np.array([f"{i + 1:06d}.SZ" for i in range(stocks)]),
        "trade_dates": np.array(
            [(start + timedelta(days=i)).date() for i in range(days)], dtype="datetime64[D]"
        ),
    }


def test_v2_trade_day_close_fields_cannot_change_signal():
    panel = _panel(days=100, stocks=25)
    expected = TrendOpenSignalV2().calc_batch(panel)
    changed = {key: value.copy() if hasattr(value, "copy") else value for key, value in panel.items()}
    for key in ("high", "low", "close", "amount"):
        changed[key][-1] = np.linspace(0.01, 999.0, 25)
    np.testing.assert_array_equal(TrendOpenSignalV2().calc_batch(changed)[-1], expected[-1])


def test_v2_is_sparse_pre_ranked_and_neutral_without_breadth():
    result = TrendOpenSignalV2().calc_batch(_panel(days=100, stocks=25))
    assert TrendOpenSignalV2.pre_ranked is True
    assert np.all(result == 0.0)


def test_v2_output_is_zero_or_bounded_quality_score():
    result = TrendOpenSignalV2().calc_batch(_panel(days=100, stocks=25))
    assert np.isfinite(result).all()
    assert np.all((result == 0.0) | ((result >= 0.5) & (result <= 1.0)))
