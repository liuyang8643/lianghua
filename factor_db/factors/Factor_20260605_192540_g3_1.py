import numpy as np

MIN_RAW_PRICE = 2.0
FIN_WEIGHT = 0.35
__thesis__ = "小盘基底乘财务质量成长放大器，优中选小非线性增强"


class Factor_20260605_192540_g3_1:
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        o = panel['open']
        bv = ~np.isnan(o) & (o >= MIN_RAW_PRICE) & ~panel['st_mask']
        n = bv.sum(axis=1, keepdims=True).clip(1)

        def _zscore(x):
            f = np.where(np.isfinite(x), x, 0.0)
            mu = (f * bv).sum(axis=1, keepdims=True) / n
            d = (f - mu) * bv
            v = (d * d).sum(axis=1, keepdims=True) / n
            return d / np.sqrt(np.maximum(v, 1e-12))

        ts = panel['total_share']
        ts_ok = np.isfinite(ts) & (ts > 0)
        ts_log = np.where(ts_ok, np.log(ts), np.nan)
        ts_log_med = np.nanmedian(ts_log, axis=1, keepdims=True)
        ts_log_med = np.where(np.isfinite(ts_log_med), ts_log_med, 18.0)
        ts_final = np.where(ts_ok, ts, np.exp(ts_log_med))
        log_mcap = np.log(np.maximum(o * ts_final, 1.0))
        small_z = -_zscore(log_mcap)

        fin_z = 0.2 * (
            _zscore(panel['roe'])
            + _zscore(panel['gross_margin'])
            + _zscore(panel['profit_yoy'])
            + _zscore(panel['revenue_yoy'])
            + _zscore(panel['eps'])
        )

        score = small_z * (1.0 + FIN_WEIGHT * np.tanh(fin_z))
        return np.where(bv & np.isfinite(score), score, np.nan)
