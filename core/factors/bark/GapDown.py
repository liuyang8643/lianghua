"""跳空缺口因子 — 向下跳空=高分(超跌买入)."""
import numpy as np
from core.factors.helpers import BaseFactor

MIN_PRICE = 2.0


class GapDown(BaseFactor):
    hist_days = 3

    def calc_batch(self, panel: dict) -> np.ndarray:
        opn = panel["open"]
        close = panel["close"]
        st = panel["st_mask"]
        valid = ~np.isnan(opn) & (opn >= MIN_PRICE) & ~st

        gap = np.full_like(opn, np.nan)
        gap[1:] = opn[1:] / close[:-1] - 1.0

        return np.where(valid, -gap, np.nan)
