"""营收增速因子 — 原始 revenue_yoy, rank归一化处理分布."""
import numpy as np
from core.factors.helpers import BaseFactor

MIN_PRICE = 2.0


class Growth(BaseFactor):
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        opn = panel["open"]
        st = panel["st_mask"]
        revenue_yoy = panel["revenue_yoy"]
        valid = ~np.isnan(opn) & (opn >= MIN_PRICE) & ~st & ~np.isnan(revenue_yoy)
        return np.where(valid, revenue_yoy, np.nan)
