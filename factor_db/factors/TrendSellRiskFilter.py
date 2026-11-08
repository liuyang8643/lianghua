"""Causal trend-break filter for the small-cap base strategy."""

import numpy as np

from factor_db.factors.TrendOpenSignalV4 import _rolling_sums, _sell_signal_v4


class TrendSellRiskFilter:
    """Reject stocks whose completed-price trend has triggered a V4 exit."""

    hist_days = 201
    pre_ranked = False
    requires_full_history = False

    def calc_batch(self, panel: dict) -> np.ndarray:
        open_ = np.asarray(panel["open"], dtype=np.float64)
        close = np.asarray(panel["close"], dtype=np.float64)
        amount = np.asarray(panel["amount"], dtype=np.float64)
        st_mask = np.asarray(panel["st_mask"], dtype=bool)
        amount_sums = _rolling_sums(amount, (20,))[0]
        price_amount_sums = _rolling_sums(close * amount, (20,))[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            maw20 = price_amount_sums / amount_sums
        sell_risk = _sell_signal_v4(
            panel, open_,
            np.asarray(panel["high"], dtype=np.float64),
            np.asarray(panel["low"], dtype=np.float64),
            close,
            amount,
            maw20,
        )
        valid = np.isfinite(open_) & (open_ > 0.0) & ~st_mask
        return np.where(valid & ~sell_risk, 1.0, np.nan).astype(np.float32)


class TrendSellRiskScore(TrendSellRiskFilter):
    """Continuous ranking form that keeps risky stocks as fallback candidates."""

    def calc_batch(self, panel: dict) -> np.ndarray:
        raw = super().calc_batch(panel)
        return np.nan_to_num(raw, nan=0.0).astype(np.float32)
