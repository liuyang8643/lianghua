"""Strict corporate-action-safe downside volatility at the tradable Open."""

from __future__ import annotations

import numpy as np


class DownsideOpenVolatilityStrict:
    """Prefer stocks with low 20-day downside Open-path volatility.

    Return T is fully known at open[T] and is adjusted with official
    preClose[T]:

        (close[T-1] / open[T-1]) * (open[T] / preClose[T]) - 1

    The factor requires exactly 20 finite returns and never fills missing
    history.  Higher scores mean lower downside volatility.
    """

    hist_days = 20
    pre_ranked = False
    requires_full_history = False

    def calc_batch(self, panel: dict) -> np.ndarray:
        open_price = np.asarray(panel["open"], dtype=np.float64)
        close = np.asarray(panel["close"], dtype=np.float64)
        pre_close = np.asarray(panel["preClose"], dtype=np.float64)
        st_mask = np.asarray(panel["st_mask"], dtype=bool)
        rows, stocks = open_price.shape
        result = np.full((rows, stocks), np.nan, dtype=np.float32)
        window = self.hist_days
        if rows <= window:
            return result

        downside_sq = np.full((rows, stocks), np.nan, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            downside_sq[1:] = close[:-1] / open_price[:-1]
            downside_sq[1:] *= open_price[1:]
            downside_sq[1:] /= pre_close[1:]
            downside_sq[1:] -= 1.0
        finite = np.isfinite(downside_sq)
        np.minimum(downside_sq, 0.0, out=downside_sq)
        np.square(downside_sq, out=downside_sq)
        downside_sq[~finite] = 0.0

        # The first valid T is row 20 and uses returns[1:21].
        counts = np.sum(finite[1:window + 1], axis=0, dtype=np.int32)
        sums_sq = np.sum(downside_sq[1:window + 1], axis=0, dtype=np.float64)
        for row in range(window, rows):
            if row > window and row % 256 == 0:
                start = row - window + 1
                counts = np.sum(
                    finite[start:row + 1], axis=0, dtype=np.int32,
                )
                sums_sq = np.sum(
                    downside_sq[start:row + 1], axis=0, dtype=np.float64,
                )
            downside_volatility = np.sqrt(np.maximum(sums_sq, 0.0) / window)
            legal = (
                (counts == window)
                & np.isfinite(downside_volatility)
                & np.isfinite(open_price[row])
                & (open_price[row] >= 2.0)
                & ~st_mask[row]
            )
            result[row, legal] = -downside_volatility[legal]

            if row + 1 < rows:
                outgoing = row - window + 1
                incoming = row + 1
                counts += finite[incoming]
                counts -= finite[outgoing]
                sums_sq += downside_sq[incoming]
                sums_sq -= downside_sq[outgoing]

        return result
