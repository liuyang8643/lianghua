"""Causal hard drawdown trend filter."""

import numpy as np

from factor_db.factors.TrendOpenSignalV4 import (
    _lag,
    _rolling_extreme_with_age,
)


class TrendHardDrawdownFilter:
    """Reject only stocks already at least 20% below a recent completed high."""

    hist_days = 21
    pre_ranked = False
    requires_full_history = False

    def calc_batch(self, panel: dict) -> np.ndarray:
        open_ = np.asarray(panel["open"], dtype=np.float64)
        high = np.asarray(panel["high"], dtype=np.float64)
        st_mask = np.asarray(panel["st_mask"], dtype=bool)
        completed_high = _lag(high, 1)
        high_price, high_age = _rolling_extreme_with_age(completed_high, 20, "max")
        with np.errstate(divide="ignore", invalid="ignore"):
            drawdown = (high_price - open_) / high_price
        hard = (high_age + 1 >= 2) & (drawdown >= 0.20)
        valid = np.isfinite(open_) & (open_ > 0.0) & ~st_mask
        return np.where(valid & ~hard, 1.0, np.nan).astype(np.float32)
