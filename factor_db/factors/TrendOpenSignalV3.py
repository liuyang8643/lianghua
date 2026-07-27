"""Pure price-path trend signal for open-auction selection."""

import numpy as np

from factor_db.factors.TrendOpenSignal import (
    _cross_confirm,
    _lag,
    _lag_bool,
    _ratio,
    _rolling_extreme_with_age,
    _rolling_mean,
    _rolling_sum,
)
from factor_db.factors.TrendOpenSignalV2 import _sell_signal_v2


MIN_RAW_OPEN = 2.0
MIN_ACTIVE_STOCKS = 20
CROSS_DAYS = (2, 3, 4, 5)
MARKET_MOMENTUM_FLOOR = -0.01375
MARKET_MA200_FLOOR = 0.987
STOP_LOSS = -0.14
TRAILING_DRAWDOWN = -0.22
MA_BREAK_MARGIN = -0.035
REVERSAL_WEIGHT = 0.25


class TrendOpenSignalV3:
    """Source-style trend state with a completed price-path ranking."""

    hist_days = 201
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
        buy_at_close = _buy_signal_v3(
            panel, open_, high, low, close, amount, maw5, maw20
        )
        buy_at_open = _lag_bool(buy_at_close, 1) & np.isfinite(open_) & (open_ > 0)
        base_sell_at_open = _sell_signal_v2(
            panel, open_, high, low, close, amount, maw20
        )

        rows = np.arange(open_.shape[0], dtype=np.int32)[:, None]
        last_buy = np.maximum.accumulate(np.where(buy_at_open, rows, -1), axis=0)
        sell_at_open = base_sell_at_open | _extra_sell_signal_v3(
            open_, high, close, maw20, last_buy, rows
        )
        last_sell = np.maximum.accumulate(np.where(sell_at_open, rows, -1), axis=0)
        holding = (last_buy > last_sell) & (last_buy >= 0)
        eligible = (
            holding & np.isfinite(open_) & (open_ >= MIN_RAW_OPEN) & ~st_mask
        )
        active = np.sum(eligible, axis=1) >= MIN_ACTIVE_STOCKS
        eligible &= active[:, None]

        completed_close = _lag(close, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            momentum120 = completed_close / _lag(close, 121) - 1.0
            completed_return = completed_close / _lag(close, 2) - 1.0
        volatility20 = np.sqrt(
            _rolling_mean(completed_return * completed_return, 20)
        )
        reversal120 = np.clip((0.40 - momentum120) / 0.80, 0.0, 1.0)
        low_vol20 = np.clip((0.06 - volatility20) / 0.05, 0.0, 1.0)
        score = (
            REVERSAL_WEIGHT * reversal120
            + (1.0 - REVERSAL_WEIGHT) * low_vol20
            + 0.01
        )

        pre_close = np.asarray(panel["preClose"], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            stock_return = close / pre_close - 1.0
        market_universe = (
            np.isfinite(stock_return) & np.isfinite(close) & ~st_mask
        )
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
            market_ma200_ratio = (
                completed_market_index / _lag(market_ma200, 1)
            )
        market_good = (
            np.isfinite(market_momentum20)
            & np.isfinite(market_ma200_ratio)
            & (market_momentum20 >= MARKET_MOMENTUM_FLOOR)
            & (market_ma200_ratio >= MARKET_MA200_FLOOR)
        )

        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        return np.where(eligible & market_good[:, None], score, 0.0).astype(
            np.float32
        )


def _extra_sell_signal_v3(open_, high, close, maw20, last_buy, rows):
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
        (age >= 2) & (peak_age + 1 >= 2)
        & (trailing_drawdown <= TRAILING_DRAWDOWN)
    )
    ma_break = (
        (completed_maw20 < previous_maw20)
        & (open_ < completed_maw20 * (1.0 + MA_BREAK_MARGIN))
        & (completed_close < completed_maw20 * (1.0 + MA_BREAK_MARGIN))
    )
    return np.isfinite(open_) & (open_ > 0) & (
        stop_loss | trailing_exit | ma_break
    )


def _buy_signal_v3(
    panel, open_, high, low, close, amount, maw5, maw20,
    *, max_avg_amount=2.5e8,
):
    with np.errstate(divide="ignore", invalid="ignore"):
        close_position = (close - low) / (high - low)
        amount_ratio = (_ratio(amount, 1) + _ratio(amount, 2)) / 2.0
        ret10 = close / _lag(close, 10) - 1.0
        body = (close - open_) / (high - low)

    valid20 = np.isfinite(maw20) & np.isfinite(close)
    universe = valid20 & ~np.asarray(panel["st_mask"], dtype=bool)
    breadth_num = np.sum(universe & (close >= maw20), axis=1)
    breadth_den = np.sum(universe, axis=1)
    breadth = np.divide(
        breadth_num,
        breadth_den,
        out=np.zeros_like(breadth_num, dtype=np.float64),
        where=breadth_den > 0,
    )[:, None]

    price_valid = (
        np.isfinite(open_)
        & np.isfinite(high)
        & np.isfinite(low)
        & np.isfinite(close)
        & np.isfinite(amount)
        & (open_ > 0)
        & (high > 0)
        & (low > 0)
        & (close > 0)
        & (amount > 0)
        & (high >= low)
    )
    amount_cap_ok = (
        True
        if max_avg_amount is None
        else _rolling_mean(amount, 3) <= max_avg_amount
    )
    common = (
        (_rolling_sum(price_valid.astype(float), 25) == 25)
        & amount_cap_ok
        & (close_position <= 0.98)
        & (breadth >= 0.43)
        & ~((ret10 <= 0.09) & (body <= -0.40))
        & (amount_ratio > 1.5)
        & (amount_ratio <= 1.825)
        & (_rolling_sum((high == low).astype(float), 3) == 0)
    )
    crosses = {
        days: _cross_confirm(close, high, low, maw5, maw20, days)
        for days in CROSS_DAYS
    }
    cross = np.logical_or.reduce(list(crosses.values()))
    changes = close - _lag(close, 1)
    path10 = _rolling_sum(np.abs(changes), 10)
    with np.errstate(divide="ignore", invalid="ignore"):
        efficiency = np.abs(close - _lag(close, 10)) / path10
        positive_ratio = _rolling_sum((changes > 0).astype(float), 10) / 10.0
        slope = maw20 / _lag(maw20, 5) - 1.0
        distance = close / maw20 - 1.0
    early_quality = (
        (path10 > 0)
        & (efficiency >= 0.60)
        & (positive_ratio >= 0.70)
        & (slope >= 0.0)
        & (distance >= 0.0)
        & (distance <= 0.08)
    )
    enough_history = np.arange(close.shape[0])[:, None] >= 39
    return enough_history & common & cross & (~crosses[4] | early_quality)
