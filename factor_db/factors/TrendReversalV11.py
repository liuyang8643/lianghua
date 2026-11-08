"""Strict completed-path variant of the V7 reversal factor."""

from __future__ import annotations

import numpy as np

from factor_db.factors.TrendOpenSignal import _lag, _rolling_mean
from factor_db.factors.TrendOpenSignalV4 import _rolling_extreme


def _rolling_max_complete(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling maximum that rejects a window containing any non-finite value."""
    finite_fraction = _rolling_mean(
        np.isfinite(values).astype(np.float64), window,
    )
    maximum = _rolling_extreme(values, window, "max")
    return np.where(finite_fraction == 1.0, maximum, np.nan)


class TrendReversalV11:
    """V7 controlled-pullback score with strict missing-history exposure."""

    hist_days = 61
    pre_ranked = False
    requires_full_history = False

    def calc_batch(self, panel: dict) -> np.ndarray:
        open_ = np.asarray(panel["open"], dtype=np.float64)
        high = np.asarray(panel["high"], dtype=np.float64)
        close = np.asarray(panel["close"], dtype=np.float64)
        st_mask = np.asarray(panel["st_mask"], dtype=bool)

        completed = _lag(close, 1)
        completed_high = _lag(high, 1)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            daily = completed / _lag(close, 2) - 1.0
            peak20 = _rolling_max_complete(completed_high, 20)
            ret20 = completed / _lag(close, 21) - 1.0
            ret60 = completed / _lag(close, 61) - 1.0
            volatility20 = np.sqrt(_rolling_mean(daily * daily, 20))
            drawdown20 = completed / peak20 - 1.0

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
        legal = np.isfinite(open_) & (open_ >= 2.0) & ~st_mask
        valid = legal & components_finite & np.isfinite(score)
        return np.where(valid, score, np.nan).astype(np.float32, copy=False)
