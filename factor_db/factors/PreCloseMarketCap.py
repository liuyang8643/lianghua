"""Point-in-time market-cap rank using the completed reference price."""

import numpy as np


class PreCloseMarketCap:
    """Negative market value based on preClose, never on T-day open."""

    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        pre_close = np.asarray(panel["preClose"], dtype=np.float64)
        total_share = np.asarray(panel["total_share"], dtype=np.float64)
        valid = (
            np.isfinite(pre_close)
            & (pre_close > 0.0)
            & np.isfinite(total_share)
            & (total_share > 0.0)
        )
        with np.errstate(over="ignore", invalid="ignore"):
            score = -(pre_close * total_share) / 1e8
        return np.where(valid, score, np.nan).astype(np.float32)
