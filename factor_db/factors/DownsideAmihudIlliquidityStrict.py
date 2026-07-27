"""Strict completed-day downside price-impact factor."""

from __future__ import annotations

import numpy as np


class DownsideAmihudIlliquidityStrict:
    """Prefer stocks with low downside price impact over 20 completed days.

    The fixed score is the negative mean of
    ``max(-(close[d] / preClose[d] - 1), 0) / (amount[d] / 1e8)`` for
    ``d in [T-20, T)``.  Every observation must be finite and have positive
    prices and amount; invalid history remains NaN.
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
        result = np.full((rows, stocks), np.nan, dtype=np.float32)
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
            official_return = close / pre_close - 1.0
            downside_impact = np.where(
                valid_daily,
                np.maximum(-official_return, 0.0) / (amount / 1e8),
                0.0,
            )

        counts = np.sum(valid_daily[:window], axis=0, dtype=np.int32)
        sums = np.sum(downside_impact[:window], axis=0, dtype=np.float64)
        for row in range(window, rows):
            averages = sums / window
            valid = (counts == window) & np.isfinite(averages)
            result[row, valid] = -averages[valid]

            if row + 1 < rows:
                outgoing = row - window
                counts += valid_daily[row]
                counts -= valid_daily[outgoing]
                sums += downside_impact[row]
                sums -= downside_impact[outgoing]

        return result
