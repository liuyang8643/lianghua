"""Risk-controlled V2 of the open-auction aggregate trend factor."""

import numpy as np

from factor_db.factors.TrendOpenSignal import (
    _buy_signal,
    _effective_rebound,
    _lag,
    _lag_bool,
    _rolling_extreme_with_age,
    _rolling_max,
    _rolling_mean,
    _rolling_min,
    _rolling_sum,
    _select_lagged,
    _variable_trailing_mean,
    compute_limit_up_matrix,
)


MIN_ACTIVE_STOCKS = 20
MIN_RAW_OPEN = 2.0


class TrendOpenSignalV2:
    """Original trend state with stricter exits and an internal breadth gate."""

    hist_days = 71
    pre_ranked = True
    requires_full_history = True

    def calc_batch(self, panel: dict) -> np.ndarray:
        open_ = np.asarray(panel["open"], dtype=np.float64)
        high = np.asarray(panel["high"], dtype=np.float64)
        low = np.asarray(panel["low"], dtype=np.float64)
        close = np.asarray(panel["close"], dtype=np.float64)
        amount = np.asarray(panel["amount"], dtype=np.float64)
        st_mask = np.asarray(panel["st_mask"], dtype=bool)

        maw5 = _rolling_mean(close * amount, 5) / _rolling_mean(amount, 5)
        maw20 = _rolling_mean(close * amount, 20) / _rolling_mean(amount, 20)
        buy_at_close = _buy_signal(panel, open_, high, low, close, amount, maw5, maw20)
        buy_at_open = _lag_bool(buy_at_close, 1) & np.isfinite(open_) & (open_ > 0)
        sell_at_open = _sell_signal_v2(panel, open_, high, low, close, amount, maw20)

        rows = np.arange(open_.shape[0], dtype=np.int32)[:, None]
        last_buy = np.maximum.accumulate(np.where(buy_at_open, rows, -1), axis=0)
        last_sell = np.maximum.accumulate(np.where(sell_at_open, rows, -1), axis=0)
        # Buy and sell now execute at the same open, so a sell vetoes a same-day buy.
        holding = (last_buy > last_sell) & (last_buy >= 0)

        eligible = holding & np.isfinite(open_) & (open_ >= MIN_RAW_OPEN) & ~st_mask
        active_count = np.sum(eligible, axis=1, keepdims=True)
        active_regime = active_count >= MIN_ACTIVE_STOCKS
        quality = _quality_score(open_, high, close, maw20)
        return np.where(eligible & active_regime, 0.5 + 0.5 * quality, 0.0).astype(np.float32)


def _sell_signal_v2(panel, today_open, high, low, close, amount, maw20):
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
    hard = (days_from_high >= 2) & (drawdown >= 0.20)
    soft = (
        (days_from_high >= 5) & (drawdown >= dynamic_limit) & ~rebound
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
    return valid_open & (
        broken
        | ((hard | soft) & enough_drawdown_history)
        | ((fast_break | confirmed_break) & enough_trend_history)
    )


def _quality_score(today_open, high, close, maw20):
    completed_close = _lag(close, 1)
    completed_maw20 = _lag(maw20, 1)
    changes = completed_close - _lag(close, 2)
    path10 = _rolling_sum(np.abs(changes), 10)
    with np.errstate(divide="ignore", invalid="ignore"):
        efficiency = np.abs(completed_close - _lag(close, 11)) / path10
        slope5 = completed_maw20 / _lag(maw20, 6) - 1.0
        distance = completed_close / completed_maw20 - 1.0
        open_to_high20 = today_open / _lag(_rolling_max(high, 20), 1)

    efficiency_q = np.clip((efficiency - 0.20) / 0.80, 0.0, 1.0)
    slope_q = np.clip((slope5 + 0.02) / 0.06, 0.0, 1.0)
    distance_q = np.clip(1.0 - np.abs(distance - 0.03) / 0.12, 0.0, 1.0)
    open_strength_q = np.clip((open_to_high20 - 0.85) / 0.15, 0.0, 1.0)
    quality = (
        0.35 * efficiency_q + 0.25 * slope_q
        + 0.20 * distance_q + 0.20 * open_strength_q
    )
    return np.where(np.isfinite(quality), quality, 0.0)
