"""Dense, stock-local trend quality for the V5 small-cap blend.

This is the individual-stock part of the V4 idea: a bounded reversal term
and a low-volatility term computed from completed price paths.  It deliberately
omits V3/V4's market breadth and synthetic market gate, and it does not read
any size, amount, or fundamental field.  The current-row open is used only for
the existing legal-price contract.
"""

from __future__ import annotations

import numpy as np

from factor_db.factors.TrendOpenSignal import _lag, _rolling_mean


class TrendIndividualV5:
    """Continuous per-stock trend quality, ranked across the full universe."""

    hist_days = 201
    pre_ranked = False
    requires_full_history = False

    def calc_batch(self, panel: dict) -> np.ndarray:
        open_ = np.asarray(panel["open"], dtype=np.float64)
        close = np.asarray(panel["close"], dtype=np.float64)
        st_mask = np.asarray(panel["st_mask"], dtype=bool)

        completed_close = _lag(close, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            momentum120 = completed_close / _lag(close, 121) - 1.0
            completed_return = completed_close / _lag(close, 2) - 1.0
        volatility20 = np.sqrt(
            _rolling_mean(completed_return * completed_return, 20)
        )
        reversal120 = np.clip((0.40 - momentum120) / 0.80, 0.0, 1.0)
        low_vol20 = np.clip((0.06 - volatility20) / 0.05, 0.0, 1.0)
        score = 0.25 * reversal120 + 0.75 * low_vol20
        score = np.nan_to_num(score, nan=0.5, posinf=0.5, neginf=0.5)

        legal = np.isfinite(open_) & (open_ >= 2.0) & ~st_mask
        return np.where(legal, score, np.nan).astype(np.float32)
