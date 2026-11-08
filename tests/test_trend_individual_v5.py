from datetime import datetime, timedelta

import numpy as np

from factor_db.factors.TrendIndividualV5 import TrendIndividualV5


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


def _copy_panel(panel):
    return {
        key: value.copy() if hasattr(value, "copy") else value
        for key, value in panel.items()
    }


def test_v5_current_non_open_fields_do_not_change_signal():
    panel = _panel()
    expected = TrendIndividualV5().calc_batch(panel)
    changed = _copy_panel(panel)
    for key in ("high", "low", "close", "amount", "volume", "preClose"):
        changed[key][-1] = np.linspace(0.01, 999.0, panel[key].shape[1])
    np.testing.assert_array_equal(TrendIndividualV5().calc_batch(changed)[-1], expected[-1])


def test_v5_open_only_changes_legal_gate():
    panel = _panel()
    expected = TrendIndividualV5().calc_batch(panel)
    changed = _copy_panel(panel)
    changed["open"][-1] *= np.linspace(0.8, 1.2, panel["open"].shape[1])
    np.testing.assert_array_equal(TrendIndividualV5().calc_batch(changed)[-1], expected[-1])
    changed["open"][-1, 0] = 1.99
    assert TrendIndividualV5().calc_batch(changed)[-1, 0] != expected[-1, 0]


def test_v5_prefix_is_independent_of_future_rows():
    panel = _panel(days=300)
    expected = TrendIndividualV5().calc_batch(panel)[:260]
    truncated = {
        key: value[:260].copy()
        if isinstance(value, np.ndarray) and value.shape[0] == 300
        else value.copy() if hasattr(value, "copy") else value
        for key, value in panel.items()
    }
    np.testing.assert_array_equal(TrendIndividualV5().calc_batch(truncated), expected)


def test_v5_has_no_size_or_amount_dependency():
    panel = _panel()
    expected = TrendIndividualV5().calc_batch(panel)
    changed = _copy_panel(panel)
    changed["amount"] *= 1024.0
    changed["volume"] *= 1024.0
    changed["total_share"] *= np.linspace(0.01, 100.0, panel["total_share"].shape[1])[None, :]
    np.testing.assert_array_equal(TrendIndividualV5().calc_batch(changed), expected)
