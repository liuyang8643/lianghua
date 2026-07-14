import numpy as np


class TrueMarketCap:
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        raw_open = panel["open"]
        total_share = panel["total_share"]

        base_valid = (
            ~np.isnan(raw_open)
            & ~np.isnan(total_share)
            & (total_share > 0)
        )

        total_mv_yi = (raw_open * total_share) / 1e8
        score = -total_mv_yi

        return np.where(base_valid, score, np.nan)
