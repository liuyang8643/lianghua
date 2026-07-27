"""Strict completed-price individual trend quality for the V11 blend."""

from __future__ import annotations

import numpy as np

from factor_db.factors.TrendOpenSignal import _lag, _rolling_mean


class TrendIndividualV5Strict:
    """120-day reversal plus 20-day low volatility, with missing history exposed."""

    hist_days = 201
    pre_ranked = False
    requires_full_history = False

    def calc_batch(self, panel: dict) -> np.ndarray:
        open_price = np.asarray(panel["open"], dtype=np.float64)
        close = np.asarray(panel["close"], dtype=np.float64)
        st_mask = np.asarray(panel["st_mask"], dtype=bool)

        completed_close = _lag(close, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            momentum120 = completed_close / _lag(close, 121) - 1.0
            completed_return = completed_close / _lag(close, 2) - 1.0
        volatility20 = np.sqrt(
            _rolling_mean(completed_return * completed_return, 20)
        )
        history_valid = np.isfinite(momentum120) & np.isfinite(volatility20)
        reversal120 = np.clip((0.40 - momentum120) / 0.80, 0.0, 1.0)
        low_vol20 = np.clip((0.06 - volatility20) / 0.05, 0.0, 1.0)
        score = 0.25 * reversal120 + 0.75 * low_vol20

        legal = (
            history_valid
            & np.isfinite(open_price)
            & (open_price >= 2.0)
            & ~st_mask
        )
        return np.where(legal, score, np.nan).astype(np.float32)
