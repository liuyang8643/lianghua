"""Strict completed-window turnover ceilings for low-liquidity style purity."""

from __future__ import annotations

import numpy as np


class _MeanAmount20CeilingFilter:
    """Mask stocks above a fixed 20-day completed average amount."""

    hist_days = 20
    max_mean_amount: float

    def calc_batch(self, panel: dict) -> np.ndarray:
        amount = np.asarray(panel["amount"], dtype=np.float64)
        rows, stocks = amount.shape
        result = np.full((rows, stocks), np.nan, dtype=np.float32)
        window = self.hist_days
        if rows <= window:
            return result

        finite = np.isfinite(amount)
        safe = np.where(finite, amount, 0.0)
        counts = np.sum(finite[:window], axis=0, dtype=np.int32)
        sums = np.sum(safe[:window], axis=0, dtype=np.float64)
        for row in range(window, rows):
            mean_amount = sums / window
            allowed = (
                (counts == window)
                & np.isfinite(mean_amount)
                & (mean_amount <= self.max_mean_amount)
            )
            result[row, allowed] = 1.0
            if row + 1 < rows:
                incoming = row
                outgoing = row - window
                counts += finite[incoming]
                counts -= finite[outgoing]
                sums += safe[incoming]
                sums -= safe[outgoing]
        return result


class FilterMeanAmount20Max40M(_MeanAmount20CeilingFilter):
    max_mean_amount = 40_000_000.0


class FilterMeanAmount20Max45M(_MeanAmount20CeilingFilter):
    max_mean_amount = 45_000_000.0


class FilterMeanAmount20Max50M(_MeanAmount20CeilingFilter):
    max_mean_amount = 50_000_000.0


class FilterMeanAmount20Max55M(_MeanAmount20CeilingFilter):
    max_mean_amount = 55_000_000.0


class FilterMeanAmount20Max60M(_MeanAmount20CeilingFilter):
    max_mean_amount = 60_000_000.0


class FilterMeanAmount20Max70M(_MeanAmount20CeilingFilter):
    max_mean_amount = 70_000_000.0


class FilterMeanAmount20Max80M(_MeanAmount20CeilingFilter):
    max_mean_amount = 80_000_000.0


class FilterMeanAmount20Max100M(_MeanAmount20CeilingFilter):
    max_mean_amount = 100_000_000.0
