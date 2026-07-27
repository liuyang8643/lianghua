from datetime import datetime, timedelta

import numpy as np

from factor_db.factors.TrendOpenSignal import (
    _cross_confirm,
    _effective_rebound as _reference_effective_rebound,
    _rolling_mean as _reference_rolling_mean,
    _rolling_extreme_with_age as _reference_extreme_with_age,
    _rolling_sum as _reference_rolling_sum,
)
from factor_db.factors.TrendOpenSignalV2 import _sell_signal_v2
from factor_db.factors.TrendOpenSignalV3 import (
    _buy_signal_v3,
    _extra_sell_signal_v3,
)
from factor_db.factors.TrendOpenSignalV4 import (
    TrendOpenSignalV4,
    _buy_signal_v4,
    _compose_v4_score,
    _cross_signals,
    _effective_rebound,
    _extra_sell_signal_v4,
    _rolling_extreme_with_age,
    _rolling_sum,
    _sell_signal_v4,
)


def _panel(days=340, stocks=25):
    turn = days // 2
    row_trend = np.concatenate((
        np.linspace(10.0, 8.5, turn, endpoint=False),
        np.linspace(8.5, 12.0, days - turn),
    ))[:, None]
    stock_scale = np.linspace(0.9, 1.1, stocks)[None, :]
    close = row_trend * stock_scale
    open_ = close * (1.0 + np.linspace(-0.005, 0.005, stocks)[None, :])
    amount = np.full((days, stocks), 1e8)
    amount *= np.linspace(0.8, 1.2, stocks)[None, :]
    amount[2::3] *= 1.6
    start = datetime(2026, 1, 1)
    return {
        "open": open_,
        "high": np.maximum(open_, close) * 1.01,
        "low": np.minimum(open_, close) * 0.99,
        "close": close,
        "volume": amount / close,
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


def test_v4_fast_rolling_helpers_match_reference_with_nan_and_ties():
    rng = np.random.default_rng(20260716)
    values = rng.normal(size=(73, 11))
    values[::7, ::3] = np.nan
    values[20:24, 4] = 3.0
    for window in (2, 5, 20, 25):
        np.testing.assert_allclose(
            _rolling_sum(values, window),
            _reference_rolling_sum(values, window),
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )
        for kind in ("max", "min"):
            actual_values, actual_ages = _rolling_extreme_with_age(
                values, window, kind
            )
            expected_values, expected_ages = _reference_extreme_with_age(
                values, window, kind
            )
            np.testing.assert_array_equal(actual_values, expected_values)
            finite = np.isfinite(expected_values)
            np.testing.assert_array_equal(actual_ages[finite], expected_ages[finite])


def test_v4_effective_rebound_matches_reference():
    rng = np.random.default_rng(20260717)
    low = rng.uniform(8.0, 10.0, size=(90, 13))
    high = low + rng.uniform(0.0, 2.0, size=low.shape)
    completed_high = np.vstack((np.full((1, 13), np.nan), high[:-1]))
    completed_low = np.vstack((np.full((1, 13), np.nan), low[:-1]))
    high_price, high_age = _reference_extreme_with_age(
        completed_high, 20, "max"
    )
    np.testing.assert_array_equal(
        _effective_rebound(completed_high, completed_low, high_age, high_price),
        _reference_effective_rebound(
            completed_high, completed_low, high_age, high_price
        ),
    )


def test_v4_fast_lifecycle_helpers_match_v3_reference():
    panel = _panel()
    open_ = panel["open"]
    high = panel["high"]
    low = panel["low"]
    close = panel["close"]
    amount = panel["amount"]
    maw5 = (
        _reference_rolling_mean(close * amount, 5)
        / _reference_rolling_mean(amount, 5)
    )
    maw20 = (
        _reference_rolling_mean(close * amount, 20)
        / _reference_rolling_mean(amount, 20)
    )
    crosses = _cross_signals(close, high, low, maw5, maw20)
    for days, actual in crosses.items():
        np.testing.assert_array_equal(
            actual, _cross_confirm(close, high, low, maw5, maw20, days)
        )
    np.testing.assert_array_equal(
        _buy_signal_v4(panel, open_, high, low, close, amount, maw5, maw20),
        _buy_signal_v3(
            panel,
            open_,
            high,
            low,
            close,
            amount,
            maw5,
            maw20,
            max_avg_amount=None,
        ),
    )
    np.testing.assert_array_equal(
        _sell_signal_v4(panel, open_, high, low, close, amount, maw20),
        _sell_signal_v2(panel, open_, high, low, close, amount, maw20),
    )
    rows = np.arange(close.shape[0], dtype=np.int32)[:, None]
    buys = _buy_signal_v4(panel, open_, high, low, close, amount, maw5, maw20)
    last_buy = np.maximum.accumulate(np.where(buys, rows, -1), axis=0)
    np.testing.assert_array_equal(
        _extra_sell_signal_v4(open_, high, close, maw20, last_buy, rows),
        _extra_sell_signal_v3(open_, high, close, maw20, last_buy, rows),
    )


def test_v4_is_sparse_finite_and_uses_one_fixed_scalar():
    result = TrendOpenSignalV4().calc_batch(_panel())
    assert TrendOpenSignalV4.pre_ranked is True
    assert np.isfinite(result).all()
    assert np.all(result[:200] == 0.0)
    assert np.count_nonzero(result[200:]) > 0
    score = _compose_v4_score(
        np.array([0.4, 0.6]), np.array([-0.01, 0.01])
    )
    assert score.shape == (2,)
    assert np.all((score > 0.0) & (score < 1.01))


def test_v4_trade_day_non_open_fields_cannot_change_signal():
    panel = _panel()
    expected = TrendOpenSignalV4().calc_batch(panel)[-1]
    assert np.count_nonzero(expected) > 0
    changed = _copy_panel(panel)
    for key in ("high", "low", "close", "volume", "amount"):
        changed[key][-1] = np.linspace(0.01, 999.0, 25)
    np.testing.assert_array_equal(TrendOpenSignalV4().calc_batch(changed)[-1], expected)


def test_v4_trade_day_open_gap_changes_score():
    panel = _panel()
    expected = TrendOpenSignalV4().calc_batch(panel)[-1]
    changed = _copy_panel(panel)
    changed["open"][-1] = changed["preClose"][-1] * 0.98
    actual = TrendOpenSignalV4().calc_batch(changed)[-1]
    assert np.count_nonzero(actual) > 0
    assert not np.array_equal(actual, expected)


def test_v4_prefix_is_independent_of_future_rows():
    panel = _panel(days=360)
    expected = TrendOpenSignalV4().calc_batch(panel)[:280]
    truncated = {}
    for key, value in panel.items():
        if isinstance(value, np.ndarray) and value.shape[0] == 360:
            truncated[key] = value[:280].copy()
        else:
            truncated[key] = value.copy() if hasattr(value, "copy") else value
    np.testing.assert_array_equal(TrendOpenSignalV4().calc_batch(truncated), expected)


def test_v4_has_no_absolute_amount_or_size_dependency():
    panel = _panel()
    expected = TrendOpenSignalV4().calc_batch(panel)
    changed = _copy_panel(panel)
    changed["amount"] *= 1024.0
    changed["volume"] *= 1024.0
    changed["total_share"] *= np.linspace(0.01, 100.0, 25)[None, :]
    np.testing.assert_array_equal(TrendOpenSignalV4().calc_batch(changed), expected)


def test_v4_excludes_st_and_low_price_at_the_open():
    panel = _panel()
    panel["st_mask"][-1, 0] = True
    panel["open"][-1, 1] = 1.99
    result = TrendOpenSignalV4().calc_batch(panel)
    assert result[-1, 0] == 0.0
    assert result[-1, 1] == 0.0
