from datetime import datetime, timedelta

import numpy as np

from factor_db.factors.TrendOpenSignalV3 import (
    TrendOpenSignalV3,
    _compose_v3_scores,
)


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
            [(start + timedelta(days=i)).date() for i in range(days)],
            dtype="datetime64[D]",
        ),
    }


def test_v3_trade_day_close_fields_cannot_change_signal():
    panel = _panel()
    expected = TrendOpenSignalV3().calc_batch(panel)
    changed = {
        key: value.copy() if hasattr(value, "copy") else value
        for key, value in panel.items()
    }
    for key in ("high", "low", "close", "amount"):
        changed[key][-1] = np.linspace(0.01, 999.0, 25)
    np.testing.assert_array_equal(
        TrendOpenSignalV3().calc_batch(changed)[-1], expected[-1]
    )


def test_v3_score_composition_applies_daily_regime_and_bounds_scores():
    shape = (3, 2)
    components = {
        "eligible": np.ones(shape, dtype=bool),
        "active_count": np.array([2, 2, 1]),
        "quality": np.full(shape, 0.5),
        "momentum20": np.full(shape, 0.5),
        "momentum60": np.full(shape, 0.5),
        "stability": np.full(shape, 0.5),
        "breadth20": np.array([0.50, 0.30, 0.50]),
        "breadth20_smoothed": np.array([0.50, 0.35, 0.25]),
        "breadth60": np.array([0.50, 0.50, 0.50]),
        "breadth20_change5": np.array([0.0, 0.0, 0.0]),
    }
    result = _compose_v3_scores(
        components,
        min_active_stocks=2,
        breadth20_on_floor=0.40,
        breadth20_off_floor=0.30,
        breadth60_floor=0.40,
        breadth20_change5_floor=-0.05,
        score_weights=(0.4, 0.3, 0.2, 0.1),
    )
    assert np.all((result[0] >= 0.05) & (result[0] <= 1.0))
    assert np.all(result[1] > 0.0)
    assert np.all(result[2] == 0.0)


def test_v3_is_sparse_pre_ranked_and_finite():
    result = TrendOpenSignalV3().calc_batch(_panel())
    assert TrendOpenSignalV3.pre_ranked is True
    assert np.isfinite(result).all()
    assert np.all((result == 0.0) | ((result >= 0.05) & (result <= 1.0)))
