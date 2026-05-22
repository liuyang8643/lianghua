import numpy as np

MIN_PRICE = 2.0


class RevenueGrowth:
    """营收增速因子。高营收增长→成长性→高收益。"""
    hist_days = 2

    def calc_batch(self, panel: dict) -> np.ndarray:
        opn = panel["open"]
        st = panel["st_mask"]
        rg = panel["revenue_yoy"]
        valid = ~np.isnan(opn) & (opn >= MIN_PRICE) & ~st
        return np.where(valid, rg, np.nan)
