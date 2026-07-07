import numpy as np

MIN_RAW_PRICE = 2.0


class TrueMarketCap:
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        raw_open = panel["close"]
        total_share = panel["total_share"]
        st_mask = panel["st_mask"]

        base_valid = (
            ~np.isnan(raw_open)
            & (raw_open >= MIN_RAW_PRICE)
            & ~np.isnan(total_share)
            & (total_share > 0)
            & ~st_mask
        )

        total_mv_yi = (raw_open * total_share) / 1e8
        score = -total_mv_yi

        return np.where(base_valid, score, np.nan)
