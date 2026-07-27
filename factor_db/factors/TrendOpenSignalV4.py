"""Candidate-local nonlinear refinement of the V3 open trend signal."""

import numpy as np

from factor_db.factors.TrendOpenSignal import (
    _lag,
    _lag_bool,
    _select_lagged,
    compute_limit_up_matrix,
)
from factor_db.factors.TrendOpenSignalV3 import (
    CROSS_DAYS,
    MA_BREAK_MARGIN,
    MARKET_MA200_FLOOR,
    MARKET_MOMENTUM_FLOOR,
    MIN_ACTIVE_STOCKS,
    MIN_RAW_OPEN,
    STOP_LOSS,
    TRAILING_DRAWDOWN,
)


REVERSAL_WEIGHT = 0.25
STABLE_CENTER = 0.525
STABLE_SCALE = 0.18
GAP_SCALE = 0.0045
GAP_WEIGHT = 0.30


def _sigmoid(values):
    clipped = np.clip(values, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _compose_v4_score(stable_raw, negative_open_gap):
    stable = _sigmoid((stable_raw - STABLE_CENTER) / STABLE_SCALE)
    gap = _sigmoid(np.clip(negative_open_gap, -0.10, 0.10) / GAP_SCALE)
    return (1.0 - GAP_WEIGHT) * stable + GAP_WEIGHT * gap + 0.01


def _rolling_sums(values, windows):
    values = np.asarray(values, dtype=np.float64, order="F")
    valid = np.isfinite(values)
    prefix = np.empty(
        (values.shape[0] + 1, values.shape[1]), dtype=np.float64, order="F"
    )
    prefix[0] = 0.0
    filled = np.array(values, copy=True, order="F")
    filled[~valid] = 0.0
    np.cumsum(filled, axis=0, out=prefix[1:])
    counts = np.empty(
        (values.shape[0] + 1, values.shape[1]), dtype=np.int32, order="F"
    )
    counts[0] = 0
    np.cumsum(valid, axis=0, dtype=np.int32, out=counts[1:])
    outputs = []
    for window in windows:
        out = np.full(values.shape, np.nan, dtype=np.float64, order="F")
        if values.shape[0] >= window:
            sums = prefix[window:] - prefix[:-window]
            n = counts[window:] - counts[:-window]
            np.copyto(out[window - 1:], sums, where=n == window)
        outputs.append(out)
    return outputs


def _rolling_sum(values, window):
    return _rolling_sums(values, (window,))[0]


def _rolling_sum_finite(values, window):
    values = np.asarray(values, dtype=np.float64, order="F")
    prefix = np.empty(
        (values.shape[0] + 1, values.shape[1]), dtype=np.float64, order="F"
    )
    prefix[0] = 0.0
    np.cumsum(values, axis=0, out=prefix[1:])
    out = np.full(values.shape, np.nan, dtype=np.float64, order="F")
    if values.shape[0] >= window:
        out[window - 1:] = prefix[window:] - prefix[:-window]
    return out


def _rolling_mean(values, window):
    return _rolling_sum(values, window) / window


def _rolling_extreme_with_age(values, window, kind):
    """O(N) block rolling extreme with the earliest-index tie rule."""
    values = np.asarray(values, dtype=np.float64)
    rows, stocks = values.shape
    result = np.full(values.shape, np.nan, dtype=np.float64)
    ages = np.full(values.shape, -1, dtype=np.int16)
    if rows < window:
        return result, ages

    fill = -np.inf if kind == "max" else np.inf
    block_count = (rows + window - 1) // window
    padded_rows = block_count * window
    blocks = np.full((block_count, window, stocks), fill, dtype=np.float64)
    blocks.reshape(padded_rows, stocks)[:rows] = np.where(
        np.isfinite(values), values, fill
    )
    prefix_values = np.empty_like(blocks)
    suffix_values = np.empty_like(blocks)
    prefix_positions = np.empty(blocks.shape, dtype=np.int16)
    suffix_positions = np.empty(blocks.shape, dtype=np.int16)

    prefix_values[:, 0] = blocks[:, 0]
    prefix_positions[:, 0] = 0
    for position in range(1, window):
        better = (
            blocks[:, position] > prefix_values[:, position - 1]
            if kind == "max"
            else blocks[:, position] < prefix_values[:, position - 1]
        )
        prefix_values[:, position] = np.where(
            better, blocks[:, position], prefix_values[:, position - 1]
        )
        prefix_positions[:, position] = np.where(
            better, position, prefix_positions[:, position - 1]
        )

    last = window - 1
    suffix_values[:, last] = blocks[:, last]
    suffix_positions[:, last] = last
    for position in range(window - 2, -1, -1):
        better_or_tied = (
            blocks[:, position] >= suffix_values[:, position + 1]
            if kind == "max"
            else blocks[:, position] <= suffix_values[:, position + 1]
        )
        suffix_values[:, position] = np.where(
            better_or_tied, blocks[:, position], suffix_values[:, position + 1]
        )
        suffix_positions[:, position] = np.where(
            better_or_tied, position, suffix_positions[:, position + 1]
        )

    ends = np.arange(window - 1, rows, dtype=np.intp)
    starts = ends - window + 1
    flat_prefix_values = prefix_values.reshape(padded_rows, stocks)
    flat_suffix_values = suffix_values.reshape(padded_rows, stocks)
    flat_prefix_positions = prefix_positions.reshape(padded_rows, stocks)
    flat_suffix_positions = suffix_positions.reshape(padded_rows, stocks)
    left_values = flat_suffix_values[starts]
    right_values = flat_prefix_values[ends]
    take_left = (
        left_values >= right_values
        if kind == "max"
        else left_values <= right_values
    )
    left_positions = (
        (starts // window * window)[:, None]
        + flat_suffix_positions[starts]
    )
    right_positions = (
        (ends // window * window)[:, None]
        + flat_prefix_positions[ends]
    )
    chosen_values = np.where(take_left, left_values, right_values)
    chosen_positions = np.where(take_left, left_positions, right_positions)
    finite = np.isfinite(chosen_values)
    result[ends] = np.where(finite, chosen_values, np.nan)
    computed_ages = ends[:, None] - chosen_positions
    ages[ends] = np.where(finite, computed_ages, -1).astype(np.int16)
    return result, ages


def _rolling_extreme(values, window, kind):
    values = np.asarray(values, dtype=np.float64)
    rows, stocks = values.shape
    result = np.full(values.shape, np.nan, dtype=np.float64)
    if rows < window:
        return result
    fill = -np.inf if kind == "max" else np.inf
    block_count = (rows + window - 1) // window
    padded_rows = block_count * window
    blocks = np.full((block_count, window, stocks), fill, dtype=np.float64)
    blocks.reshape(padded_rows, stocks)[:rows] = np.where(
        np.isfinite(values), values, fill
    )
    operation = np.maximum if kind == "max" else np.minimum
    prefix = operation.accumulate(blocks, axis=1)
    suffix = operation.accumulate(blocks[:, ::-1], axis=1)[:, ::-1]
    ends = np.arange(window - 1, rows, dtype=np.intp)
    starts = ends - window + 1
    result[ends] = operation(
        suffix.reshape(padded_rows, stocks)[starts],
        prefix.reshape(padded_rows, stocks)[ends],
    )
    result[~np.isfinite(result)] = np.nan
    return result


def _variable_trailing_mean(values, lengths, max_length):
    values = np.asarray(values, dtype=np.float64, order="F")
    valid_value = np.isfinite(values)
    prefix_sum = np.empty(
        (values.shape[0] + 1, values.shape[1]), dtype=np.float64, order="F"
    )
    prefix_sum[0] = 0.0
    filled = np.array(values, copy=True, order="F")
    filled[~valid_value] = 0.0
    np.cumsum(filled, axis=0, out=prefix_sum[1:])
    prefix_count = np.empty(
        (values.shape[0] + 1, values.shape[1]), dtype=np.int32, order="F"
    )
    prefix_count[0] = 0
    np.cumsum(
        valid_value, axis=0, dtype=np.int32, out=prefix_count[1:]
    )
    end = np.arange(1, values.shape[0] + 1, dtype=np.intp)[:, None]
    valid_length = (
        (lengths >= 1) & (lengths <= max_length) & (lengths <= end)
    )
    start = np.clip(end - lengths, 0, values.shape[0])
    columns = np.arange(values.shape[1], dtype=np.intp)[None, :]
    sums = prefix_sum[end, columns] - prefix_sum[start, columns]
    counts = prefix_count[end, columns] - prefix_count[start, columns]
    means = np.divide(
        sums,
        counts,
        out=np.full_like(sums, np.nan),
        where=counts > 0,
    )
    return np.where(valid_length & (counts == lengths), means, np.nan)


def _effective_rebound(
    high, low, high_age, high_price, candidate_mask=None
):
    """Evaluate the rebound only over each cell's actual post-high window."""
    result = np.zeros(high.shape, dtype=bool)
    if candidate_mask is None:
        candidate_mask = np.ones(high.shape, dtype=bool)
    for age in range(1, 20):
        selected_rows, selected_cols = np.nonzero(
            (high_age == age) & candidate_mask
        )
        if not len(selected_rows):
            continue
        chronological_offsets = np.arange(age - 1, -1, -1, dtype=np.intp)
        low_rows = selected_rows[:, None] - chronological_offsets[None, :]
        lows = low[low_rows, selected_cols[:, None]]
        low_positions = np.argmin(lows, axis=1)
        post_low = lows[np.arange(len(lows)), low_positions]
        low_offsets = chronological_offsets[low_positions]
        rebound_high = np.full(len(lows), np.nan, dtype=np.float64)
        for offset in range(age):
            members = np.flatnonzero(low_offsets == offset)
            if not len(members):
                continue
            offsets = np.arange(offset, -1, -1, dtype=np.intp)
            high_rows = selected_rows[members, None] - offsets[None, :]
            rebound_high[members] = np.max(
                high[high_rows, selected_cols[members, None]], axis=1
            )
        selected_high = high_price[selected_rows, selected_cols]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (rebound_high - post_low) / (selected_high - post_low)
        result[selected_rows, selected_cols] = (
            (post_low < selected_high) & (ratio >= 0.30)
        )
    return result


def _cross_signals(close, high, low, fast, slow):
    relation = (close >= slow) & (fast >= slow)
    trend = relation.copy()
    high_extreme = high.copy()
    low_extreme = low.copy()
    signals = {}
    for offset in range(1, max(CROSS_DAYS)):
        trend[offset:] &= relation[:-offset]
        high_extreme[offset:] = np.fmax(high_extreme[offset:], high[:-offset])
        low_extreme[offset:] = np.fmin(low_extreme[offset:], low[:-offset])
        days = offset + 1
        if days not in CROSS_DAYS:
            continue
        signal = np.zeros(close.shape, dtype=bool)
        start = days + 1
        with np.errstate(divide="ignore", invalid="ignore"):
            signal[start:] = (
                (fast[:-start] < slow[:-start])
                & (fast[1:-days] >= slow[1:-days])
                & trend[start:]
                & (high_extreme[start:] / close[1:-days] - 1.0 <= 0.15)
                & (1.0 - low_extreme[start:] / close[1:-days] <= 0.025)
                & (slow[start:] / slow[1:-days] - 1.0 <= 0.05)
            )
        signals[days] = signal
    return signals


def _buy_signal_v4(panel, open_, high, low, close, amount, maw5, maw20):
    with np.errstate(divide="ignore", invalid="ignore"):
        close_position = (close - low) / (high - low)
        amount_ratio = (
            amount / _lag(amount, 1) + amount / _lag(amount, 2)
        ) / 2.0
        ret10 = close / _lag(close, 10) - 1.0
        body = (close - open_) / (high - low)

    valid20 = np.isfinite(maw20) & np.isfinite(close)
    universe = valid20 & ~np.asarray(panel["st_mask"], dtype=bool)
    breadth_num = np.sum(universe & (close >= maw20), axis=1)
    breadth_den = np.sum(universe, axis=1)
    breadth = np.divide(
        breadth_num,
        breadth_den,
        out=np.zeros(breadth_num.shape, dtype=np.float64),
        where=breadth_den > 0,
    )[:, None]

    price_valid = (
        np.isfinite(open_)
        & np.isfinite(high)
        & np.isfinite(low)
        & np.isfinite(close)
        & np.isfinite(amount)
        & (open_ > 0.0)
        & (high > 0.0)
        & (low > 0.0)
        & (close > 0.0)
        & (amount > 0.0)
        & (high >= low)
    )
    common = (
        (_rolling_sum_finite(price_valid.astype(np.float64), 25) == 25)
        & (close_position <= 0.98)
        & (breadth >= 0.43)
        & ~((ret10 <= 0.09) & (body <= -0.40))
        & (amount_ratio > 1.5)
        & (amount_ratio <= 1.825)
        & (_rolling_sum_finite((high == low).astype(np.float64), 3) == 0)
    )
    crosses = _cross_signals(close, high, low, maw5, maw20)
    cross = np.logical_or.reduce(list(crosses.values()))
    changes = close - _lag(close, 1)
    path10, positive10 = (
        _rolling_sum(np.abs(changes), 10),
        _rolling_sum_finite((changes > 0.0).astype(np.float64), 10),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        efficiency = np.abs(close - _lag(close, 10)) / path10
        positive_ratio = positive10 / 10.0
        slope = maw20 / _lag(maw20, 5) - 1.0
        distance = close / maw20 - 1.0
    early_quality = (
        (path10 > 0.0)
        & (efficiency >= 0.60)
        & (positive_ratio >= 0.70)
        & (slope >= 0.0)
        & (distance >= 0.0)
        & (distance <= 0.08)
    )
    enough_history = np.arange(close.shape[0])[:, None] >= 39
    return enough_history & common & cross & (~crosses[4] | early_quality)


def _sell_signal_v4(panel, today_open, high, low, close, amount, maw20):
    valid_open = np.isfinite(today_open) & (today_open > 0.0)
    limit_up = compute_limit_up_matrix(panel)
    target = _lag(limit_up, 1) * (1.0 - 0.002)
    broken = (_lag(high, 1) >= target) & (today_open <= target / 1.04)

    completed_high = _lag(high, 1)
    completed_low = _lag(low, 1)
    completed_close = _lag(close, 1)
    completed_amount = _lag(amount, 1)
    high_price, high_age = _rolling_extreme_with_age(
        completed_high, 20, "max"
    )
    days_from_high = high_age + 1
    rise_low = _select_lagged(
        _rolling_extreme(completed_low, 21, "min"), high_age
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        rise_ratio = high_price / rise_low - 1.0
        drawdown = (high_price - today_open) / high_price
    dynamic_limit = np.minimum(
        0.20, np.maximum(0.045, 0.045 + rise_ratio * 0.20)
    )
    hard = (days_from_high >= 2) & (drawdown >= 0.20)
    soft_candidate = (
        (days_from_high >= 5)
        & (drawdown >= dynamic_limit)
        & (completed_close < high_price * (1.0 - dynamic_limit * 0.70))
    )
    rebound = _effective_rebound(
        completed_high,
        completed_low,
        high_age,
        high_price,
        candidate_mask=soft_candidate,
    )
    soft = soft_candidate & ~rebound

    stage_high, stage_age = _rolling_extreme_with_age(
        completed_high, 25, "max"
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        retreat = (stage_high - today_open) / stage_high
    amount_from_high = _variable_trailing_mean(
        completed_amount, stage_age + 1, 25
    )
    amount_before_high = _select_lagged(
        _rolling_mean(completed_amount, 25), stage_age + 1
    )
    maw_t1 = _lag(maw20, 1)
    maw_t2 = _lag(maw20, 2)
    falling_maw = maw_t1 < maw_t2
    fast_break = (
        falling_maw
        & (retreat >= 0.05)
        & (stage_age + 1 <= 3)
        & (today_open < maw_t1)
        & (amount_from_high >= amount_before_high)
    )
    confirmed_break = (
        falling_maw
        & (_lag(close, 2) < maw_t2)
        & (completed_close < maw_t1)
        & (today_open < maw_t1)
    )
    enough_drawdown_history = np.arange(today_open.shape[0])[:, None] >= 40
    enough_trend_history = np.arange(today_open.shape[0])[:, None] >= 70
    return valid_open & (
        broken
        | ((hard | soft) & enough_drawdown_history)
        | ((fast_break | confirmed_break) & enough_trend_history)
    )


def _extra_sell_signal_v4(open_, high, close, maw20, last_buy, rows):
    safe_buy_rows = np.clip(last_buy, 0, open_.shape[0] - 1)
    entry_price = np.take_along_axis(open_, safe_buy_rows, axis=0)
    entry_price = np.where(last_buy >= 0, entry_price, np.nan)
    age = rows - last_buy
    completed_high = _lag(high, 1)
    completed_close = _lag(close, 1)
    completed_maw20 = _lag(maw20, 1)
    previous_maw20 = _lag(maw20, 2)
    peak, peak_age = _rolling_extreme_with_age(completed_high, 30, "max")
    with np.errstate(divide="ignore", invalid="ignore"):
        entry_return = open_ / entry_price - 1.0
        trailing_drawdown = open_ / peak - 1.0
    stop_loss = (age >= 1) & (entry_return <= STOP_LOSS)
    trailing_exit = (
        (age >= 2)
        & (peak_age + 1 >= 2)
        & (trailing_drawdown <= TRAILING_DRAWDOWN)
    )
    ma_break = (
        (completed_maw20 < previous_maw20)
        & (open_ < completed_maw20 * (1.0 + MA_BREAK_MARGIN))
        & (completed_close < completed_maw20 * (1.0 + MA_BREAK_MARGIN))
    )
    return np.isfinite(open_) & (open_ > 0.0) & (
        stop_loss | trailing_exit | ma_break
    )


class TrendOpenSignalV4:
    """V3 trend lifecycle ranked by fixed stable-trend and T-open transforms."""

    hist_days = 201
    pre_ranked = True
    requires_full_history = False

    def calc_batch(self, panel: dict) -> np.ndarray:
        open_ = np.asarray(panel["open"], dtype=np.float64)
        high = np.asarray(panel["high"], dtype=np.float64)
        low = np.asarray(panel["low"], dtype=np.float64)
        close = np.asarray(panel["close"], dtype=np.float64)
        amount = np.asarray(panel["amount"], dtype=np.float64)
        st_mask = np.asarray(panel["st_mask"], dtype=bool)

        amount_sums = _rolling_sums(amount, (5, 20))
        price_amount_sums = _rolling_sums(close * amount, (5, 20))
        with np.errstate(divide="ignore", invalid="ignore"):
            maw5 = price_amount_sums[0] / amount_sums[0]
            maw20 = price_amount_sums[1] / amount_sums[1]
        buy_at_close = _buy_signal_v4(
            panel, open_, high, low, close, amount, maw5, maw20
        )
        valid_open = np.isfinite(open_) & (open_ > 0.0)
        buy_at_open = _lag_bool(buy_at_close, 1) & valid_open
        base_sell_at_open = _sell_signal_v4(
            panel, open_, high, low, close, amount, maw20
        )

        rows = np.arange(open_.shape[0], dtype=np.int32)[:, None]
        last_buy = np.maximum.accumulate(np.where(buy_at_open, rows, -1), axis=0)
        sell_at_open = base_sell_at_open | _extra_sell_signal_v4(
            open_, high, close, maw20, last_buy, rows
        )
        last_sell = np.maximum.accumulate(np.where(sell_at_open, rows, -1), axis=0)
        holding = (last_buy > last_sell) & (last_buy >= 0)
        eligible = holding & valid_open & (open_ >= MIN_RAW_OPEN) & ~st_mask
        eligible &= (np.sum(eligible, axis=1) >= MIN_ACTIVE_STOCKS)[:, None]

        completed_close = _lag(close, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            momentum120 = completed_close / _lag(close, 121) - 1.0
            completed_return = completed_close / _lag(close, 2) - 1.0
        volatility20 = np.sqrt(
            _rolling_mean(completed_return * completed_return, 20)
        )
        reversal120 = np.clip((0.40 - momentum120) / 0.80, 0.0, 1.0)
        low_vol20 = np.clip((0.06 - volatility20) / 0.05, 0.0, 1.0)
        stable_raw = (
            REVERSAL_WEIGHT * reversal120
            + (1.0 - REVERSAL_WEIGHT) * low_vol20
        )

        pre_close = np.asarray(panel["preClose"], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            stock_return = close / pre_close - 1.0

        market_universe = np.isfinite(stock_return) & np.isfinite(close) & ~st_mask
        clipped_return = np.where(
            market_universe, np.clip(stock_return, -0.20, 0.20), 0.0
        )
        market_count = np.sum(market_universe, axis=1)
        market_return = np.divide(
            np.sum(clipped_return, axis=1),
            market_count,
            out=np.zeros(market_count.shape, dtype=np.float64),
            where=market_count > 0,
        )
        market_index = np.cumprod(1.0 + market_return)
        completed_market_index = _lag(market_index, 1)
        market_ma200 = _rolling_mean(market_index[:, None], 200)[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            market_momentum20 = (
                completed_market_index / _lag(market_index, 21) - 1.0
            )
            market_ma200_ratio = completed_market_index / _lag(market_ma200, 1)
        market_good = (
            np.isfinite(market_momentum20)
            & np.isfinite(market_ma200_ratio)
            & (market_momentum20 >= MARKET_MOMENTUM_FLOOR)
            & (market_ma200_ratio >= MARKET_MA200_FLOOR)
        )

        selected = (
            eligible
            & market_good[:, None]
            & np.isfinite(stable_raw)
            & np.isfinite(pre_close)
            & (pre_close > 0.0)
        )
        score = np.zeros(open_.shape, dtype=np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            negative_open_gap = -(
                open_[selected] / pre_close[selected] - 1.0
            )
        score[selected] = _compose_v4_score(
            stable_raw[selected], negative_open_gap
        ).astype(np.float32)
        return score
