"""流动性过滤 — 每日剔除近20日均成交额最低10%的股票。返回 1.0(通过)/NaN(剔除)"""

import numpy as np

MIN_RAW_PRICE = 2.0
MIN_AMOUNT_PCT = 10  # 剔除底部百分位


class LiquidityFilter:
    hist_days = 20

    # 记录最近一次计算的阈值，供外部查询
    thresholds: list[float] = []

    def calc_batch(self, panel: dict) -> np.ndarray:
        amount = panel["amount"]
        st_mask = panel["st_mask"]
        close = panel["close"]

        amount_known = np.empty_like(amount)
        amount_known[0] = np.nan
        amount_known[1:] = amount[:-1]

        w = self.hist_days
        filled = np.where(np.isnan(amount_known), 0.0, amount_known)
        cum_sum = np.cumsum(filled, axis=0)
        cum_n = np.cumsum(~np.isnan(amount_known), axis=0).astype(float)
        avg_amount = np.empty_like(amount, dtype=float)
        avg_amount[:w] = np.nan
        avg_amount[w:] = (cum_sum[w:] - cum_sum[:-w]) / (cum_n[w:] - cum_n[:-w])

        price_ok = ~np.isnan(close) & (close >= MIN_RAW_PRICE) & ~st_mask
        result = np.full_like(avg_amount, np.nan)

        thresholds = []
        for d in range(w, avg_amount.shape[0]):
            row_avg = avg_amount[d]
            row_valid = price_ok[d] & ~np.isnan(row_avg)
            if not row_valid.any():
                thresholds.append(0.0)
                continue
            threshold = float(np.percentile(row_avg[row_valid], MIN_AMOUNT_PCT))
            thresholds.append(threshold)
            result[d] = np.where(row_valid & (row_avg >= threshold), 1.0, np.nan)

        LiquidityFilter.thresholds = thresholds
        return result

