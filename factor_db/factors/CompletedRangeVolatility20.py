"""Strict 20-day completed intraday-range volatility factor."""

from __future__ import annotations

import numpy as np


class CompletedRangeVolatility20:
    """Prefer stocks with small, stable completed-day price ranges.

    Row T is scored from ``high/low`` observations in ``[T-20, T)``.  The
    current open is only a legality gate; no T-day HLCVA field or gap return is
    part of the signal.
    """

    hist_days = 20
    pre_ranked = False
    requires_full_history = False

    def calc_batch(self, panel: dict) -> np.ndarray:
        high = np.asarray(panel["high"], dtype=np.float64)
        low = np.asarray(panel["low"], dtype=np.float64)
        open_ = np.asarray(panel["open"], dtype=np.float64)
        st_mask = np.asarray(panel["st_mask"], dtype=bool)
        if high.shape != low.shape or high.shape != open_.shape:
            raise ValueError("high, low, and open must have matching shapes")
        if st_mask.shape != high.shape:
            raise ValueError("st_mask must match the price panel shape")

        rows, stocks = high.shape
        result = np.full((rows, stocks), np.nan, dtype=np.float32)
        window = self.hist_days
        if rows <= window:
            return result

        valid_range = (
            np.isfinite(high)
            & np.isfinite(low)
            & (high > 0.0)
            & (low > 0.0)
            & (high >= low)
        )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            log_range = np.where(valid_range, np.log(high / low), 0.0)

        counts = np.sum(valid_range[:window], axis=0, dtype=np.int32)
        sums = np.sum(log_range[:window], axis=0, dtype=np.float64)
        for row in range(window, rows):
            mean_range = sums / window
            valid = (
                (counts == window)
                & np.isfinite(mean_range)
                & np.isfinite(open_[row])
                & (open_[row] >= 2.0)
                & ~st_mask[row]
            )
            result[row, valid] = -mean_range[valid]

            if row + 1 < rows:
                incoming = row
                outgoing = row - window
                counts += valid_range[incoming]
                counts -= valid_range[outgoing]
                sums += log_range[incoming]
                sums -= log_range[outgoing]

        return result
