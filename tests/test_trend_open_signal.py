import numpy as np
from datetime import datetime, timedelta

from core.backtest import _compute_factor_scores
from factor_db.factors.TrendOpenSignal import TrendOpenSignal


def _panel(days=50, stocks=3):
    base = np.linspace(8.0, 10.0, days)[:, None] * np.ones((1, stocks))
    amount = np.full_like(base, 1e8)
    start = datetime(2026, 1, 1)
    return {
        "open": base.copy(), "high": base * 1.01, "low": base * 0.99,
        "close": base.copy(), "amount": amount,
        "st_mask": np.zeros_like(base, dtype=bool),
        "preClose": np.vstack((base[:1], base[:-1])),
        "issue_price": np.full(stocks, 8.0),
        "stock_codes": np.array([f"00000{i + 1}.SZ" for i in range(stocks)]),
        "trade_dates": np.array(
            [(start + timedelta(days=i)).date() for i in range(days)], dtype="datetime64[D]"
        ),
    }


def test_trade_day_close_fields_cannot_change_signal():
    panel = _panel()
    factor = TrendOpenSignal()
    expected = factor.calc_batch(panel)
    changed = {key: value.copy() for key, value in panel.items()}
    for key in ("high", "low", "close", "amount"):
        changed[key][-1] = np.array([999.0, np.nan, 0.01])
    np.testing.assert_array_equal(factor.calc_batch(changed)[-1], expected[-1])


def test_empty_signal_is_a_neutral_pre_ranked_score():
    result = TrendOpenSignal().calc_batch(_panel())
    assert TrendOpenSignal.pre_ranked is True
    assert np.all(result == 0.0)


def test_signal_is_shifted_to_next_open():
    panel = _panel(days=3, stocks=1)
    # Test the execution-time shift independently of the strategy thresholds.
    panel["open"][-1, 0] = np.nan
    result = TrendOpenSignal().calc_batch(panel)
    assert result.shape == panel["open"].shape
    assert result[-1, 0] == 0.0


def test_standalone_signal_does_not_fill_topn_with_zero_scores():
    panel = _panel()
    start = datetime(2026, 1, 1)
    result = _compute_factor_scores(
        [start + timedelta(days=49)], panel["stock_codes"].tolist(),
        {"TrendOpenSignal": 1.0}, [TrendOpenSignal], data=panel,
    )
    masks = result[2]
    assert "_standalone_sparse_signal" in masks
    assert not masks["_standalone_sparse_signal"][-1].any()
