"""Strict, corporate-action-safe Amihud illiquidity factor."""

from __future__ import annotations

import numpy as np
from factor.library.completed_windows import completed_finite_window_sum


class AmihudIlliquidityStrict:
    """Prefer illiquid stocks using 20 complete, official-return observations.

    Row ``T`` uses only rows ``[T-20, T)``.  Each completed daily return is
    ``close[d] / preClose[d] - 1`` so ex-rights reference prices are handled by
    the authoritative runtime field.  Any missing, non-finite, or non-positive
    input invalidates the entire window instead of being averaged away.
    """

    hist_days = 20
    pre_ranked = False
    requires_full_history = False

    def calc_batch(self, panel: dict) -> np.ndarray:
        close = np.asarray(panel["close"], dtype=np.float64)
        pre_close = np.asarray(panel["preClose"], dtype=np.float64)
        amount = np.asarray(panel["amount"], dtype=np.float64)
        if close.shape != pre_close.shape or close.shape != amount.shape:
            raise ValueError("close, preClose, and amount must have matching shapes")

        rows, stocks = close.shape
        result = np.full((rows, stocks), np.nan, dtype=np.float64)
        window = self.hist_days
        if rows <= window:
            return result

        valid_daily = (
            np.isfinite(close)
            & (close > 0.0)
            & np.isfinite(pre_close)
            & (pre_close > 0.0)
            & np.isfinite(amount)
            & (amount > 0.0)
        )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            daily = np.where(
                valid_daily,
                np.abs(close / pre_close - 1.0) / (amount / 1e8),
                0.0,
            )

        valid_daily &= np.isfinite(daily)
        values = np.where(valid_daily, daily, np.nan)
        return completed_finite_window_sum(values, window) / window
