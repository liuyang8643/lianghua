import numpy as np

W_PROFIT = 0.35
W_ROE = 0.30
W_REV = 0.20
W_SIZE = 0.15

MIN_RAW_PRICE = 2.0

__thesis__ = "盈利增速与质量叠加小盘偏好，多维信号互补增强截面区分"


class Factor_20260605_192540_g1_0:
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        raw_open = panel['open']
        st_mask = panel['st_mask']
        total_share = panel['total_share']
        profit_yoy = panel['profit_yoy']
        roe = panel['roe']
        revenue_yoy = panel['revenue_yoy']

        base_valid = ~np.isnan(raw_open) & (raw_open >= MIN_RAW_PRICE) & ~st_mask

        def _zscore(x):
            med = np.nanmedian(x, axis=1, keepdims=True)
            mad = np.nanmedian(np.abs(x - med), axis=1, keepdims=True)
            z = (x - med) / (mad * 1.4826 + 1e-12)
            return np.where(np.isnan(z), 0.0, z.astype(np.float64))

        z_profit = _zscore(profit_yoy)
        z_roe = _zscore(roe)
        z_rev = _zscore(revenue_yoy)

        ts_med = np.nanmedian(total_share, axis=1, keepdims=True)
        ts_med = np.where(np.isnan(ts_med) | (ts_med <= 0), 1e8, ts_med)
        ts = np.where(np.isnan(total_share) | (total_share <= 0), ts_med, total_share)
        log_mcap = -np.log(np.maximum(raw_open * ts / 1e8, 1e-6))
        z_size = _zscore(log_mcap)

        jitter = np.arange(raw_open.shape[1], dtype=np.float64) * 1e-12
        score = W_PROFIT * z_profit + W_ROE * z_roe + W_REV * z_rev + W_SIZE * z_size + jitter

        return np.where(base_valid & np.isfinite(score), score, np.nan)
