"""盈利质量因子 — ROE + 毛利率原始混合."""
import numpy as np
from core.factors.helpers import BaseFactor

MIN_PRICE = 2.0


class Profitability(BaseFactor):
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        opn = panel["open"]
        st = panel["st_mask"]
        roe = panel["roe"]
        gross_margin = panel["gross_margin"]
        valid = ~np.isnan(opn) & (opn >= MIN_PRICE) & ~st
        score = (roe - 15.0) / 5.0 + (gross_margin - 40.0) / 10.0
        return np.where(valid, score, np.nan)
