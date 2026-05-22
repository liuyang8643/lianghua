"""ROC变动率因子 — 原始值."""
import numpy as np
from core.factors.helpers import BaseFactor

MIN_PRICE = 2.0


class ROC(BaseFactor):
    hist_days = 14

    def calc_batch(self, panel: dict) -> np.ndarray:
        close = panel["close"]
        opn = panel["open"]
        st = panel["st_mask"]
        valid = ~np.isnan(opn) & (opn >= MIN_PRICE) & ~st

        cp = np.roll(close, 1, axis=0); cp[0] = np.nan
        cp12 = np.roll(cp, 12, axis=0); cp12[:12] = np.nan
        roc = np.full_like(close, np.nan)
        roc[13:] = (cp[13:] - cp12[13:]) / np.where(np.abs(cp12[13:]) > 1e-9, cp12[13:], np.nan) * 100.0

        return np.where(valid, roc, np.nan)
