"""Strict prior-day close-location strength."""

from __future__ import annotations

import numpy as np


class LaggedCloseLocationStrength:
    """Prefer stocks whose completed close finished high in its daily range.

    At decision day T the score is:

        (close[T-1] - low[T-1]) / (high[T-1] - low[T-1])

    Missing or zero-range completed bars remain missing.  The current Open is
    used only for the existing legal-price contract.
    """

    hist_days = 1
    pre_ranked = False
    requires_full_history = False

    def calc_batch(self, panel: dict) -> np.ndarray:
        open_price = np.asarray(panel["open"], dtype=np.float64)
        high = np.asarray(panel["high"], dtype=np.float64)
        low = np.asarray(panel["low"], dtype=np.float64)
        close = np.asarray(panel["close"], dtype=np.float64)
        st_mask = np.asarray(panel["st_mask"], dtype=bool)
        score = np.full(open_price.shape, np.nan, dtype=np.float64)
        if len(score) < 2:
            return score.astype(np.float32)

        completed_high = high[:-1]
        completed_low = low[:-1]
        completed_close = close[:-1]
        spread = completed_high - completed_low
        valid_completed = (
            np.isfinite(completed_high)
            & np.isfinite(completed_low)
            & np.isfinite(completed_close)
            & (spread > 0.0)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            completed_score = (
                (completed_close - completed_low) / spread
            )
        legal = (
            np.isfinite(open_price[1:])
            & (open_price[1:] >= 2.0)
            & ~st_mask[1:]
        )
        valid = valid_completed & legal & np.isfinite(completed_score)
        score[1:] = np.where(valid, completed_score, np.nan)
        return score.astype(np.float32)
