"""估值因子 — EP(1/PE) + BP(1/PB) 原始值."""
import numpy as np
from core.factors.helpers import BaseFactor

MIN_PRICE = 2.0


class Valuation(BaseFactor):
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        opn = panel["open"]
        eps = panel["eps"]
        bps = panel.get("bps", None)
        st = panel["st_mask"]
        valid = ~np.isnan(opn) & (opn >= MIN_PRICE) & ~st

        ep = np.full_like(opn, np.nan)
        eps_ok = (eps > 0) & ~np.isnan(eps) & valid
        ep[eps_ok] = eps[eps_ok] / opn[eps_ok]

        if bps is not None:
            bp = np.full_like(opn, np.nan)
            bps_ok = (bps > 0) & ~np.isnan(bps) & valid
            bp[bps_ok] = bps[bps_ok] / opn[bps_ok]
            score = ep + bp
        else:
            score = ep

        return np.where(valid, score, np.nan)
