"""Point-in-time market capitalization without a T-open gap signal."""

from __future__ import annotations

import numpy as np


def _lag(values: np.ndarray) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=np.float64)
    result[1:] = values[:-1]
    return result


class TrueMarketCapT1:
    """Prefer smaller companies using only completed T-1 information."""

    hist_days = 1
    pre_ranked = False
    requires_full_history = False

    def calc_batch(self, panel: dict) -> np.ndarray:
        completed_close = _lag(np.asarray(panel["close"], dtype=np.float64))
        known_shares = _lag(np.asarray(panel["total_share"], dtype=np.float64))
        with np.errstate(invalid="ignore"):
            market_cap = completed_close * known_shares
        valid = (
            np.isfinite(completed_close)
            & np.isfinite(known_shares)
            & (completed_close > 0.0)
            & (known_shares > 0.0)
        )
        return np.where(valid, -market_cap / 1e8, np.nan).astype(np.float32)
