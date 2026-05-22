import numpy as np

MIN_PRICE = 2.0


class CashFlowYield:
    """经营现金流收益率。operating_cf_ps/open，高现金流回报→高质量→高收益。"""
    hist_days = 2

    def calc_batch(self, panel: dict) -> np.ndarray:
        opn = panel["open"]
        st = panel["st_mask"]
        ocf = panel["operating_cf_ps"]
        valid = ~np.isnan(opn) & (opn >= MIN_PRICE) & ~st & ~np.isnan(ocf) & (ocf > 0)
        cf_yield = ocf / np.where(opn > 0, opn, np.nan)
        return np.where(valid, cf_yield, np.nan)
