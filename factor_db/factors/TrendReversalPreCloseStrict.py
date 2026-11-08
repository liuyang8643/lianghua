"""Strict controlled-pullback factor based on official completed-day returns."""

from __future__ import annotations

import numpy as np


_SHORT_WINDOW = 20
_LONG_WINDOW = 60


class TrendReversalPreCloseStrict:
    """V7-style pullback quality without raw-close or missing-data tolerance.

    Row ``T`` uses official-return paths ending at ``T-1``.  Multi-day returns
    compound ``close[d] / preClose[d]`` and the drawdown component compares the
    completed adjusted close wealth with adjusted completed-day highs.  The
    current open and ST state are legality gates only.

    Rolling state is updated one completed day at a time, with every update
    vectorized across the full stock cross-section.  Prefix rings preserve the
    exact summation order of a full-matrix ``cumsum`` while avoiding repeated
    full-panel rolling passes.
    """

    hist_days = _LONG_WINDOW
    pre_ranked = False
    requires_full_history = False

    def calc_batch(self, panel: dict) -> np.ndarray:
        open_ = np.asarray(panel["open"], dtype=np.float64)
        high = np.asarray(panel["high"], dtype=np.float64)
        close = np.asarray(panel["close"], dtype=np.float64)
        pre_close = np.asarray(panel["preClose"], dtype=np.float64)
        st_mask = np.asarray(panel["st_mask"], dtype=bool)
        if not (open_.shape == high.shape == close.shape == pre_close.shape):
            raise ValueError("open, high, close, and preClose must have matching shapes")
        if st_mask.shape != close.shape:
            raise ValueError("st_mask must match the price panel shape")

        rows, stocks = close.shape
        valid_close = (
            np.isfinite(close)
            & (close > 0.0)
            & np.isfinite(pre_close)
            & (pre_close > 0.0)
        )
        valid_high = valid_close & np.isfinite(high) & (high > 0.0) & (high >= close)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            close_gross = np.where(valid_close, close / pre_close, np.nan)
            daily_return = close_gross - 1.0
            log_close_gross = np.where(valid_close, np.log(close_gross), np.nan)
            log_high_gross = np.where(
                valid_high,
                np.log(high / pre_close),
                np.nan,
            )
            squared_daily_return = daily_return * daily_return

        finite_log_return = np.isfinite(log_close_gross)
        finite_squared_return = np.isfinite(squared_daily_return)
        safe_log_return = np.where(finite_log_return, log_close_gross, 0.0)
        safe_squared_return = np.where(
            finite_squared_return,
            squared_daily_return,
            0.0,
        )
        # This deliberately keys off valid_close, rather than finite_log_return,
        # to retain the original strict overflow behavior in adjusted wealth.
        wealth_log_return = np.where(valid_close, log_close_gross, 0.0)

        result = np.full(close.shape, np.nan, dtype=np.float32)
        cumulative_log_sum = np.zeros(stocks, dtype=np.float64)
        cumulative_squared_sum = np.zeros(stocks, dtype=np.float64)
        cumulative_log_count = np.zeros(stocks, dtype=np.int32)
        cumulative_squared_count = np.zeros(stocks, dtype=np.int32)

        # A window+1 ring ensures the prefix being subtracted cannot be
        # overwritten by the current completed day before it is consumed.
        log_prefix_ring = np.zeros((_LONG_WINDOW + 1, stocks), dtype=np.float64)
        squared_prefix_ring = np.zeros(
            (_SHORT_WINDOW + 1, stocks),
            dtype=np.float64,
        )
        log_count_ring = np.zeros((_LONG_WINDOW + 1, stocks), dtype=np.int32)
        squared_count_ring = np.zeros(
            (_SHORT_WINDOW + 1, stocks),
            dtype=np.int32,
        )

        adjusted_high_ring = np.full(
            (_SHORT_WINDOW, stocks),
            -np.inf,
            dtype=np.float64,
        )
        adjusted_high_finite_ring = np.zeros(
            (_SHORT_WINDOW, stocks),
            dtype=bool,
        )
        adjusted_high_count = np.zeros(stocks, dtype=np.int16)
        completed_log_wealth = np.zeros(stocks, dtype=np.float64)

        for row in range(1, rows):
            completed_day = row - 1

            high_slot = completed_day % _SHORT_WINDOW
            adjusted_high_count -= adjusted_high_finite_ring[high_slot]
            adjusted_log_high = (
                completed_log_wealth + log_high_gross[completed_day]
            )
            adjusted_high_is_finite = np.isfinite(adjusted_log_high)
            adjusted_high_ring[high_slot] = np.where(
                adjusted_high_is_finite,
                adjusted_log_high,
                -np.inf,
            )
            adjusted_high_finite_ring[high_slot] = adjusted_high_is_finite
            adjusted_high_count += adjusted_high_is_finite
            completed_log_wealth += wealth_log_return[completed_day]

            cumulative_log_sum += safe_log_return[completed_day]
            cumulative_squared_sum += safe_squared_return[completed_day]
            cumulative_log_count += finite_log_return[completed_day]
            cumulative_squared_count += finite_squared_return[completed_day]

            if completed_day >= _SHORT_WINDOW:
                short_prefix_slot = (
                    completed_day - _SHORT_WINDOW
                ) % (_LONG_WINDOW + 1)
                short_squared_slot = (
                    completed_day - _SHORT_WINDOW
                ) % (_SHORT_WINDOW + 1)
                short_log_sum = (
                    cumulative_log_sum - log_prefix_ring[short_prefix_slot]
                )
                short_squared_sum = (
                    cumulative_squared_sum
                    - squared_prefix_ring[short_squared_slot]
                )
                short_squared_count = (
                    cumulative_squared_count
                    - squared_count_ring[short_squared_slot]
                )
            else:
                short_log_sum = cumulative_log_sum
                short_squared_sum = cumulative_squared_sum
                short_squared_count = cumulative_squared_count

            if completed_day >= _LONG_WINDOW:
                long_prefix_slot = (
                    completed_day - _LONG_WINDOW
                ) % (_LONG_WINDOW + 1)
                long_log_sum = (
                    cumulative_log_sum - log_prefix_ring[long_prefix_slot]
                )
                long_log_count = (
                    cumulative_log_count - log_count_ring[long_prefix_slot]
                )
            else:
                long_log_sum = cumulative_log_sum
                long_log_count = cumulative_log_count

            log_slot = completed_day % (_LONG_WINDOW + 1)
            squared_slot = completed_day % (_SHORT_WINDOW + 1)
            log_prefix_ring[log_slot] = cumulative_log_sum
            log_count_ring[log_slot] = cumulative_log_count
            squared_prefix_ring[squared_slot] = cumulative_squared_sum
            squared_count_ring[squared_slot] = cumulative_squared_count

            if row < _LONG_WINDOW:
                continue

            peak20 = np.max(adjusted_high_ring, axis=0)
            with np.errstate(over="ignore", invalid="ignore"):
                ret20 = np.expm1(short_log_sum)
                ret60 = np.expm1(long_log_sum)
                volatility20 = np.sqrt(short_squared_sum / _SHORT_WINDOW)
                drawdown20 = np.expm1(completed_log_wealth - peak20)

            components_finite = (
                np.isfinite(ret20)
                & np.isfinite(ret60)
                & np.isfinite(volatility20)
                & np.isfinite(drawdown20)
            )
            score = (
                0.45 * np.clip((-ret20 + 0.20) / 0.40, 0.0, 1.0)
                + 0.25 * np.clip((-ret60 + 0.40) / 0.80, 0.0, 1.0)
                + 0.15 * np.clip((0.08 - volatility20) / 0.08, 0.0, 1.0)
                + 0.15 * np.clip((drawdown20 + 0.35) / 0.35, 0.0, 1.0)
            )
            legal = (
                np.isfinite(open_[row])
                & (open_[row] >= 2.0)
                & ~st_mask[row]
            )
            valid = (
                legal
                & (long_log_count == _LONG_WINDOW)
                & (short_squared_count == _SHORT_WINDOW)
                & (adjusted_high_count == _SHORT_WINDOW)
                & components_finite
                & np.isfinite(score)
            )
            result[row] = np.where(valid, score, np.nan)

        return result
