import numpy as np

MIN_PRICE = 2.0


class GrossMargin:
    """毛利率因子。高毛利→定价权→高质量→高收益。"""
    hist_days = 2

    def calc_batch(self, panel: dict) -> np.ndarray:
        opn = panel["open"]
        st = panel["st_mask"]
        gm = panel["gross_margin"]
        valid = ~np.isnan(opn) & (opn >= MIN_PRICE) & ~st
        return np.where(valid, gm, np.nan)
