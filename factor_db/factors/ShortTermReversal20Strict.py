"""Strict 20-day short-term reversal from completed official returns."""

from __future__ import annotations

import numpy as np


class ShortTermReversal20Strict:
    """Prefer recent losers using exactly 20 completed official returns.

    Row ``T`` is ``-expm1(sum(log(close[d] / preClose[d])))`` for
    ``d in [T-20, T)``.  Every completed close and official pre-close must be
    finite and strictly positive; otherwise the whole affected window remains
    NaN.  The current open and ST state are legality gates only.
    """

    hist_days = 20
    pre_ranked = False
    requires_full_history = False

    def calc_batch(self, panel: dict) -> np.ndarray:
        open_ = np.asarray(panel["open"], dtype=np.float64)
        close = np.asarray(panel["close"], dtype=np.float64)
        pre_close = np.asarray(panel["preClose"], dtype=np.float64)
        st_mask = np.asarray(panel["st_mask"], dtype=bool)
        if close.ndim != 2:
            raise ValueError("price panels must be two-dimensional")
        if not (open_.shape == close.shape == pre_close.shape == st_mask.shape):
            raise ValueError(
                "open, close, preClose, and st_mask must have matching shapes"
            )

        rows, stocks = close.shape
        result = np.full((rows, stocks), np.nan, dtype=np.float32)
        window = self.hist_days
        if rows <= window:
            return result

        valid_input = (
            np.isfinite(close)
            & (close > 0.0)
            & np.isfinite(pre_close)
            & (pre_close > 0.0)
        )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            gross_return = close / pre_close
            log_return = np.log(gross_return)
        valid_daily = valid_input & np.isfinite(log_return)
        safe_log_return = np.where(valid_daily, log_return, 0.0)

        rolling_count = np.sum(valid_daily[:window], axis=0, dtype=np.int32)
        rolling_log_sum = np.sum(
            safe_log_return[:window],
            axis=0,
            dtype=np.float64,
        )
        score = np.empty(stocks, dtype=np.float64)
        for row in range(window, rows):
            with np.errstate(invalid="ignore", over="ignore"):
                np.expm1(rolling_log_sum, out=score)
            score *= -1.0
            legal = (
                np.isfinite(open_[row])
                & (open_[row] >= 2.0)
                & ~st_mask[row]
            )
            valid = (
                legal
                & (rolling_count == window)
                & np.isfinite(score)
            )
            result[row, valid] = score[valid]

            if row + 1 < rows:
                outgoing = row - window
                rolling_count += valid_daily[row]
                rolling_count -= valid_daily[outgoing]
                rolling_log_sum += safe_log_return[row]
                rolling_log_sum -= safe_log_return[outgoing]

        return result
