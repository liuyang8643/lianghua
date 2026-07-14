"""Matrix replica of the exported aggregate trend strategy at the open."""

import numpy as np

from core.legality import compute_limit_up_matrix


class TrendOpenSignal:
    """1 while the exported buy/sell state machine is holding, otherwise 0."""

    hist_days = 71
    pre_ranked = True
    requires_full_history = True

    def calc_batch(self, panel: dict) -> np.ndarray:
        open_ = np.asarray(panel["open"], dtype=np.float64)
        high = np.asarray(panel["high"], dtype=np.float64)
        low = np.asarray(panel["low"], dtype=np.float64)
        close = np.asarray(panel["close"], dtype=np.float64)
        amount = np.asarray(panel["amount"], dtype=np.float64)

        maw5 = _rolling_mean(close * amount, 5) / _rolling_mean(amount, 5)
        maw20 = _rolling_mean(close * amount, 20) / _rolling_mean(amount, 20)
        buy_at_close = _buy_signal(panel, open_, high, low, close, amount, maw5, maw20)
        buy_at_open = _lag_bool(buy_at_close, 1) & np.isfinite(open_) & (open_ > 0)
        sell_at_open = _sell_signal(panel, open_, high, low, close, amount, maw20)

        # Original order is sell at the open, then buy after that day's close.
        # After moving buys to the next open, a same-open buy takes precedence.
        rows = np.arange(open_.shape[0], dtype=np.int32)[:, None]
        last_buy = np.maximum.accumulate(np.where(buy_at_open, rows, -1), axis=0)
        last_sell = np.maximum.accumulate(np.where(sell_at_open, rows, -1), axis=0)
        holding = (last_buy >= last_sell) & (last_buy >= 0)
        return holding.astype(np.float32)


def _buy_signal(panel, open_, high, low, close, amount, maw5, maw20):
    avg_amount3 = _rolling_mean(amount, 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        close_position = (close - low) / (high - low)
        amount_ratio = (_ratio(amount, 1) + _ratio(amount, 2)) / 2.0
        ret10 = close / _lag(close, 10) - 1.0
        body = (close - open_) / (high - low)

    valid20 = np.isfinite(maw20) & np.isfinite(close)
    universe = valid20 & ~np.asarray(panel["st_mask"], dtype=bool)
    breadth_num = np.sum(universe & (close >= maw20), axis=1)
    breadth_den = np.sum(universe, axis=1)
    breadth = np.divide(breadth_num, breadth_den, out=np.zeros_like(breadth_num, dtype=float), where=breadth_den > 0)[:, None]

    price_valid = (
        np.isfinite(open_) & np.isfinite(high) & np.isfinite(low) & np.isfinite(close)
        & np.isfinite(amount) & (open_ > 0) & (high > 0) & (low > 0)
        & (close > 0) & (amount > 0) & (high >= low)
    )
    common = (
        (_rolling_sum(price_valid.astype(float), 25) == 25)
        & (avg_amount3 <= 2.5e8) & (close_position <= 0.98) & (breadth >= 0.43)
        & ~((ret10 <= 0.09) & (body <= -0.40))
        & (amount_ratio > 1.5) & (amount_ratio <= 1.825)
        & (_rolling_sum((high == low).astype(float), 3) == 0)
    )
    normal = _cross_confirm(close, high, low, maw5, maw20, 5)
    early = _cross_confirm(close, high, low, maw5, maw20, 4)
    changes = close - _lag(close, 1)
    path10 = _rolling_sum(np.abs(changes), 10)
    with np.errstate(divide="ignore", invalid="ignore"):
        efficiency = np.abs(close - _lag(close, 10)) / path10
        positive_ratio = _rolling_sum((changes > 0).astype(float), 10) / 10.0
        slope = maw20 / _lag(maw20, 5) - 1.0
        distance = close / maw20 - 1.0
    early_quality = (
        (path10 > 0) & (efficiency >= 0.60) & (positive_ratio >= 0.70)
        & (slope >= 0.0) & (distance >= 0.0) & (distance <= 0.08)
    )
    enough_history = np.arange(close.shape[0])[:, None] >= 39
    return enough_history & common & (normal | (early & early_quality))


def _sell_signal(panel, today_open, high, low, close, amount, maw20):
    valid_open = np.isfinite(today_open) & (today_open > 0)
    limit_up = compute_limit_up_matrix(panel)
    target = _lag(limit_up, 1) * (1.0 - 0.002)
    broken = (_lag(high, 1) >= target) & (today_open <= target / 1.04)

    completed_high = _lag(high, 1)
    completed_low = _lag(low, 1)
    completed_close = _lag(close, 1)
    completed_amount = _lag(amount, 1)
    high_price, high_age = _rolling_extreme_with_age(completed_high, 20, "max")
    days_from_high = high_age + 1
    rise_low = _select_lagged(_rolling_min(completed_low, 21), high_age)
    with np.errstate(divide="ignore", invalid="ignore"):
        rise_ratio = high_price / rise_low - 1.0
        drawdown = (high_price - today_open) / high_price
    dynamic_limit = np.minimum(0.20, np.maximum(0.045, 0.045 + rise_ratio * 0.20))
    rebound = _effective_rebound(completed_high, completed_low, high_age, high_price)
    hard = (days_from_high >= 2) & (drawdown >= 0.20) & rebound
    soft = (
        (days_from_high == 5) & (drawdown >= dynamic_limit) & ~rebound
        & (completed_close < high_price * (1.0 - dynamic_limit * 0.70))
    )

    stage_high, stage_age = _rolling_extreme_with_age(completed_high, 25, "max")
    with np.errstate(divide="ignore", invalid="ignore"):
        retreat = (stage_high - today_open) / stage_high
    amount_from_high = _variable_trailing_mean(completed_amount, stage_age + 1, 25)
    amount_before_high = _select_lagged(_rolling_mean(completed_amount, 25), stage_age + 1)
    maw_t1 = _lag(maw20, 1)
    maw_t2 = _lag(maw20, 2)
    falling_maw = maw_t1 < maw_t2
    fast_break = (
        falling_maw & (retreat >= 0.05) & (stage_age + 1 <= 3) & (today_open < maw_t1)
        & (amount_from_high >= amount_before_high)
    )
    confirmed_break = (
        falling_maw & (_lag(close, 2) < maw_t2) & (completed_close < maw_t1)
        & (today_open < maw_t1)
    )
    enough_drawdown_history = np.arange(today_open.shape[0])[:, None] >= 40
    enough_trend_history = np.arange(today_open.shape[0])[:, None] >= 70
    return valid_open & (broken | ((hard | soft) & enough_drawdown_history) | ((fast_break | confirmed_break) & enough_trend_history))


def _cross_confirm(close, high, low, fast, slow, days):
    cross = (_lag(fast, days + 1) < _lag(slow, days + 1)) & (_lag(fast, days) >= _lag(slow, days))
    trend = _rolling_sum(((close >= slow) & (fast >= slow)).astype(float), days) == days
    base = _lag(close, days)
    with np.errstate(divide="ignore", invalid="ignore"):
        high_rise = _rolling_max(high, days) / base - 1.0
        low_drop = 1.0 - _rolling_min(low, days) / base
        slow_rise = slow / _lag(slow, days) - 1.0
    return cross & trend & (high_rise <= 0.15) & (low_drop <= 0.025) & (slow_rise <= 0.05)


def _effective_rebound(high, low, high_age, high_price):
    result = np.zeros_like(high, dtype=bool)
    if high.shape[0] < 20:
        return result
    pad = ((19, 0), (0, 0))
    high_windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(high, pad, constant_values=np.nan), 20, axis=0)
    low_windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(low, pad, constant_values=np.nan), 20, axis=0)
    positions = np.arange(20)[None, None, :]
    for start in range(0, high.shape[0], 128):
        stop = min(start + 128, high.shape[0])
        ages = high_age[start:stop]
        post_mask = positions >= (20 - ages[..., None])
        lows = np.where(post_mask, low_windows[start:stop], np.inf)
        post_low = np.min(lows, axis=2)
        low_pos = np.argmin(lows, axis=2)
        rebound_mask = positions >= low_pos[..., None]
        rebound_high = np.max(
            np.where(rebound_mask, high_windows[start:stop], -np.inf), axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = ((rebound_high - post_low)
                     / (high_price[start:stop] - post_low))
        valid = (ages >= 1) & (ages <= 19)
        result[start:stop] = valid & (post_low < high_price[start:stop]) & (ratio >= 0.30)
    return result


def _rolling_extreme_with_age(values, window, kind):
    extreme = _rolling_max(values, window) if kind == "max" else _rolling_min(values, window)
    ages = np.full(values.shape, -1, dtype=np.int16)
    if values.shape[0] >= window:
        fill = -np.inf if kind == "max" else np.inf
        safe_values = np.where(np.isfinite(values), values, fill)
        view = np.lib.stride_tricks.sliding_window_view(safe_values, window, axis=0)
        pos = np.argmax(view, axis=2) if kind == "max" else np.argmin(view, axis=2)
        ages[window - 1:] = window - 1 - pos
    return extreme, ages


def _select_lagged(values, lags):
    rows = np.arange(values.shape[0])[:, None] - lags
    valid = (lags >= 0) & (rows >= 0)
    safe_rows = np.clip(rows, 0, values.shape[0] - 1)
    selected = np.take_along_axis(values, safe_rows, axis=0)
    return np.where(valid, selected, np.nan)


def _variable_trailing_mean(values, lengths, max_length):
    valid_value = np.isfinite(values)
    prefix_sum = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(np.where(valid_value, values, 0.0), axis=0)))
    prefix_count = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(valid_value, axis=0)))
    end = np.arange(1, values.shape[0] + 1)[:, None]
    valid_length = (lengths >= 1) & (lengths <= max_length) & (lengths <= end)
    start = np.clip(end - lengths, 0, values.shape[0])
    sums = prefix_sum[end, np.arange(values.shape[1])[None, :]] - prefix_sum[start, np.arange(values.shape[1])[None, :]]
    counts = prefix_count[end, np.arange(values.shape[1])[None, :]] - prefix_count[start, np.arange(values.shape[1])[None, :]]
    means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    return np.where(valid_length & (counts == lengths), means, np.nan)


def _lag(values, periods):
    out = np.full_like(values, np.nan, dtype=np.float64)
    out[periods:] = values[:-periods]
    return out


def _lag_bool(values, periods):
    out = np.zeros_like(values, dtype=bool)
    out[periods:] = values[:-periods]
    return out


def _ratio(values, periods):
    with np.errstate(divide="ignore", invalid="ignore"):
        return values / _lag(values, periods)


def _rolling_sum(values, window):
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)
    prefix = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(filled, axis=0)))
    counts = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(valid, axis=0)))
    out = np.full_like(values, np.nan, dtype=np.float64)
    if values.shape[0] >= window:
        sums = prefix[window:] - prefix[:-window]
        n = counts[window:] - counts[:-window]
        out[window - 1:] = np.where(n == window, sums, np.nan)
    return out


def _rolling_mean(values, window):
    return _rolling_sum(values, window) / window


def _rolling_max(values, window):
    return _rolling_extreme(values, window, np.nanmax)


def _rolling_min(values, window):
    return _rolling_extreme(values, window, np.nanmin)


def _rolling_extreme(values, window, reducer):
    out = np.full_like(values, np.nan, dtype=np.float64)
    if values.shape[0] >= window:
        fill = -np.inf if reducer is np.nanmax else np.inf
        safe_values = np.where(np.isfinite(values), values, fill)
        view = np.lib.stride_tricks.sliding_window_view(safe_values, window, axis=0)
        reduced = np.max(view, axis=2) if reducer is np.nanmax else np.min(view, axis=2)
        out[window - 1:] = np.where(np.isfinite(reduced), reduced, np.nan)
    return out
