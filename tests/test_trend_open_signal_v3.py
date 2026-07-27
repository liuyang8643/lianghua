from datetime import datetime, timedelta

import numpy as np

from factor_db.factors.TrendOpenSignalV3 import TrendOpenSignalV3


def _panel(days=300, stocks=25):
    turn = days // 2
    row_trend = np.concatenate((
        np.linspace(10.0, 8.5, turn, endpoint=False),
        np.linspace(8.5, 12.0, days - turn),
    ))[:, None]
    stock_scale = np.linspace(0.9, 1.1, stocks)[None, :]
    close = row_trend * stock_scale
    open_ = close * (1.0 + np.linspace(-0.005, 0.005, stocks)[None, :])
    amount_scale = np.linspace(0.8, 1.2, stocks)[None, :]
    amount = np.full((days, stocks), 1e8) * amount_scale
    amount[2::3] *= 1.6
    volume = amount / close
    start = datetime(2026, 1, 1)
    return {
        "open": open_,
        "high": np.maximum(open_, close) * 1.01,
        "low": np.minimum(open_, close) * 0.99,
        "close": close,
        "volume": volume,
        "amount": amount,
        "total_share": np.full((days, stocks), 2e8),
        "st_mask": np.zeros_like(close, dtype=bool),
        "preClose": np.vstack((close[:1], close[:-1])),
        "issue_price": np.full(stocks, 8.0),
        "stock_codes": np.array([f"{i + 1:06d}.SZ" for i in range(stocks)]),
        "trade_dates": np.array(
            [(start + timedelta(days=i)).date() for i in range(days)],
            dtype="datetime64[D]",
        ),
    }


def _copy_panel(panel):
    return {
        key: value.copy() if hasattr(value, "copy") else value
        for key, value in panel.items()
    }


def test_v3_trade_day_close_fields_cannot_change_signal():
    panel = _panel()
    expected = TrendOpenSignalV3().calc_batch(panel)
    assert np.count_nonzero(expected) > 0
    assert np.count_nonzero(expected[-1]) == 25
    changed = _copy_panel(panel)
    for key in ("high", "low", "close", "volume", "amount", "preClose"):
        changed[key][-1] = np.linspace(0.01, 999.0, 25)
    np.testing.assert_array_equal(
        TrendOpenSignalV3().calc_batch(changed)[-1], expected[-1]
    )


def test_v3_prefix_is_independent_of_future_rows():
    panel = _panel(days=340, stocks=25)
    expected = TrendOpenSignalV3().calc_batch(panel)[:260]
    truncated = {}
    for key, value in panel.items():
        if isinstance(value, np.ndarray) and value.shape[0] == 340:
            truncated[key] = value[:260].copy()
        else:
            truncated[key] = value.copy() if hasattr(value, "copy") else value
    np.testing.assert_array_equal(TrendOpenSignalV3().calc_batch(truncated), expected)


def test_v3_is_invariant_to_stock_column_order():
    panel = _panel()
    permutation = np.arange(25)[::-1]
    reordered = {}
    for key, value in panel.items():
        if isinstance(value, np.ndarray) and value.ndim == 2:
            reordered[key] = value[:, permutation]
        elif isinstance(value, np.ndarray) and value.shape == (25,):
            reordered[key] = value[permutation]
        else:
            reordered[key] = value.copy() if hasattr(value, "copy") else value
    expected = TrendOpenSignalV3().calc_batch(panel)
    actual = TrendOpenSignalV3().calc_batch(reordered)
    np.testing.assert_array_equal(actual[:, permutation], expected)


def test_v3_is_sparse_finite_and_reaches_a_nonzero_state():
    result = TrendOpenSignalV3().calc_batch(_panel())
    assert TrendOpenSignalV3.pre_ranked is True
    assert np.isfinite(result).all()
    assert np.all(result[:200] == 0.0)
    assert np.count_nonzero(result[200:]) > 0


def test_v3_excludes_st_and_low_price_at_the_open():
    panel = _panel()
    panel["st_mask"][-1, 0] = True
    panel["open"][-1, 1] = 1.99
    result = TrendOpenSignalV3().calc_batch(panel)
    assert result[-1, 0] == 0.0
    assert result[-1, 1] == 0.0


def test_v3_does_not_use_share_count_or_other_size_inputs():
    panel = _panel()
    expected = TrendOpenSignalV3().calc_batch(panel)
    panel["total_share"] *= np.linspace(0.01, 100.0, 25)[None, :]
    np.testing.assert_array_equal(TrendOpenSignalV3().calc_batch(panel), expected)
