"""Dense stock-local reversal/repair quality for the small-cap blend."""

from __future__ import annotations

import numpy as np

from factor_db.factors.TrendOpenSignal import _lag, _rolling_mean
from factor_db.factors.TrendOpenSignalV4 import _rolling_extreme


class TrendReversalV7:
    """Rank stocks by controlled pullback quality using completed paths only.

    The score favors a recoverable pullback rather than a severely damaged
    stock: moderate 20/60-day weakness, low recent volatility, and a small
    drawdown from the completed 20-day high.  The current open is used only by
    the existing legal-price contract.
    """

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
        daily = completed / _lag(close, 2) - 1.0
        peak20 = _rolling_extreme(completed_high, 20, "max")
        with np.errstate(divide="ignore", invalid="ignore"):
            ret20 = completed / _lag(close, 21) - 1.0
            ret60 = completed / _lag(close, 61) - 1.0
            volatility20 = np.sqrt(_rolling_mean(daily * daily, 20))
            drawdown20 = completed / peak20 - 1.0

        score = (
            0.45 * np.clip((-ret20 + 0.20) / 0.40, 0.0, 1.0)
            + 0.25 * np.clip((-ret60 + 0.40) / 0.80, 0.0, 1.0)
            + 0.15 * np.clip((0.08 - volatility20) / 0.08, 0.0, 1.0)
            + 0.15 * np.clip((drawdown20 + 0.35) / 0.35, 0.0, 1.0)
        )
        score = np.nan_to_num(score, nan=0.5, posinf=0.5, neginf=0.5)
        legal = np.isfinite(open_) & (open_ >= 2.0) & ~st_mask
        return np.where(legal, score, np.nan).astype(np.float32)
