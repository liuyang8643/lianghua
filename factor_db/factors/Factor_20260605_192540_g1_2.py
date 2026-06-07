import numpy as np

SIZE_W = 0.40
QUALITY_W = 0.35
IPO_W = 0.25

__thesis__ = "盈利质量与发行溢价共振的小盘增强因子"


class Factor_20260605_192540_g1_2:
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        base_valid = ~np.isnan(panel['open']) & (panel['open'] >= 2.0) & ~panel['st_mask']

        def _zscore(x):
            x = x.astype(np.float64)
            mu = np.nanmean(x, axis=1, keepdims=True)
            sd = np.nanstd(x, axis=1, keepdims=True)
            return np.where(sd > 1e-12, (x - mu) / sd, 0.0).astype(np.float32)

        with np.errstate(divide='ignore', invalid='ignore'):
            ts = np.where(np.isfinite(panel['total_share']) & (panel['total_share'] > 0), panel['total_share'], np.nan)
            ts_fill = np.nanmedian(ts, axis=1, keepdims=True)
            ts_fill = np.where(np.isfinite(ts_fill), ts_fill, 1.0)
            mcap = panel['open'] * np.where(np.isfinite(ts), ts, ts_fill) / 1e8

            ey = panel['eps'] / panel['open']
            cf_yield = panel['operating_cf_ps'] / panel['open']
            accrual_gap = (panel['operating_cf_ps'] - panel['eps']) / panel['open']
            ip = np.where(np.isfinite(panel['issue_price']) & (panel['issue_price'] > 1e-8), panel['issue_price'], np.nan)
            ipo_ratio = panel['open'] / ip

        ey_imp = np.where(np.isfinite(ey), ey, 0.0)
        cf_imp = np.where(np.isfinite(cf_yield), cf_yield, 0.0)
        accrual_imp = np.where(np.isfinite(accrual_gap), accrual_gap, 0.0)
        roe_s = np.sign(panel['roe']) * np.abs(panel['roe']) ** (1.0 / 3.0)
        roe_imp = np.where(np.isfinite(roe_s), roe_s, 0.0)

        log_mcap = np.log(np.maximum(mcap, 1e-6))

        ipo_fill = np.nanmedian(ipo_ratio, axis=1, keepdims=True)
        ipo_fill = np.where(np.isfinite(ipo_fill), ipo_fill, 1.0)
        ipo_imp = np.where(np.isfinite(ipo_ratio) & (ipo_ratio > 0.01), ipo_ratio, np.maximum(ipo_fill, 0.01))
        ipo_score = -np.log(np.maximum(ipo_imp, 0.01))

        z_size = _zscore(-log_mcap)
        z_ey = _zscore(ey_imp)
        z_cf = _zscore(cf_imp)
        z_accrual = _zscore(accrual_imp)
        z_roe = _zscore(roe_imp)
        z_ipo = _zscore(ipo_score)

        quality_raw = z_roe + z_cf + z_accrual
        quality_gate = 0.5 + 0.5 * np.tanh(quality_raw)

        score = SIZE_W * z_size * quality_gate + QUALITY_W * z_ey * quality_gate + IPO_W * z_ipo

        return np.where(base_valid & np.isfinite(score), score, np.nan)
