import numpy as np

MIN_RAW_PRICE = 2.0


class TMC_GARP_Broad:
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        raw_open = panel["open"]
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

        fields = {"profit_yoy": 0.20, "eps": 0.10, "operating_cf_ps": 0.05, "revenue_yoy": 0.05}
        for field, weight in fields.items():
            if field == "eps":
                fin_arr = panel["eps"] / np.where(raw_open > 0, raw_open, np.nan)
            else:
                fin_arr = panel[field]
            fin_clean = np.where(base_valid & ~np.isnan(fin_arr), fin_arr, 0.0)
            score += weight * fin_clean

        return np.where(base_valid, score, np.nan)
