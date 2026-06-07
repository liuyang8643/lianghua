import numpy as np

GROWTH_W = 0.50
QUALITY_W = 0.28
VALUE_W = 0.20
TIE_W = 0.02

GROWTH_S = 2.0
QUALITY_S = 1.5
VALUE_S = 2.5

__thesis__ = "盈利成长质量估值三维GARP精选因子"


class Factor_20260605_192540_g3_0:
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        base_valid = ~np.isnan(panel['open']) & (panel['open'] >= 2.0) & ~panel['st_mask']

        def _cross_zscore(x):
            x = x.astype(np.float64)
            mu = np.nanmean(x, axis=1, keepdims=True)
            sd = np.nanstd(x, axis=1, keepdims=True)
            filled = np.where(np.isfinite(x), x, mu)
            return np.where(sd > 1e-12, (filled - mu) / sd, 0.0).astype(np.float32)

        g_profit = _cross_zscore(panel['profit_yoy'])
        g_revenue = _cross_zscore(panel['revenue_yoy'])
        growth_raw = g_profit + g_revenue
        growth_sig = np.tanh(growth_raw * GROWTH_S)

        with np.errstate(divide='ignore', invalid='ignore'):
            accrual = (panel['operating_cf_ps'] - panel['eps']) / panel['open']
        z_accrual = _cross_zscore(accrual)
        z_roe = _cross_zscore(panel['roe'])
        z_gm = _cross_zscore(panel['gross_margin'])
        quality_raw = z_accrual + z_roe + z_gm
        quality_sig = np.tanh(quality_raw * QUALITY_S)

        with np.errstate(divide='ignore', invalid='ignore'):
            ey = panel['eps'] / panel['open']
        z_ey = _cross_zscore(ey)
        value_sig = np.tanh(z_ey * VALUE_S)

        with np.errstate(divide='ignore', invalid='ignore'):
            log_mcap = np.log(np.maximum(panel['open'] * panel['total_share'] / 1e8, 1e-8))
        z_mcap = _cross_zscore(log_mcap)

        score = (GROWTH_W * growth_sig
                 + QUALITY_W * quality_sig
                 + VALUE_W * value_sig
                 + TIE_W * z_mcap)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
