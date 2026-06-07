import numpy as np

SIZE_W = 0.50
QUALITY_W = 0.28
VALUE_W = 0.22

QUALITY_S = 2.5
VALUE_S = 2.0

__thesis__ = "质量与价值双重过滤的小市值因子"


class Factor_20260605_192540_g2_1:
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        base_valid = ~np.isnan(panel['open']) & (panel['open'] >= 2.0) & ~panel['st_mask']

        def _zscore(x):
            x = x.astype(np.float64)
            mu = np.nanmean(x, axis=1, keepdims=True)
            sd = np.nanstd(x, axis=1, keepdims=True)
            return np.where(sd > 1e-12, (x - mu) / sd, np.nan).astype(np.float32)

        def _rank_norm(x):
            x = x.astype(np.float64)
            nan = np.isnan(x)
            order = np.argsort(np.argsort(np.where(nan, np.inf, x), axis=1), axis=1).astype(np.float64)
            n = (~nan).sum(axis=1, keepdims=True).astype(np.float64)
            r = 2.0 * order / np.maximum(n - 1.0, 1.0) - 1.0
            return np.where(nan, np.nan, r).astype(np.float32)

        with np.errstate(divide='ignore', invalid='ignore'):
            log_mv = np.log(panel['open'] * panel['total_share'] / 1e8)
            ey = panel['eps'] / panel['open']
            cf_yield = panel['operating_cf_ps'] / panel['open']

        roe_fill = np.where(np.isfinite(panel['roe']), panel['roe'], 0.0)
        gm_fill = np.where(np.isfinite(panel['gross_margin']), panel['gross_margin'], 0.0)
        ey_fill = np.where(np.isfinite(ey), ey, 0.0)
        cf_fill = np.where(np.isfinite(cf_yield), cf_yield, 0.0)

        size_sig = _rank_norm(-log_mv)

        roe_s = np.sign(roe_fill) * np.abs(roe_fill) ** (1.0 / 3.0)
        gm_s = np.sign(gm_fill) * np.abs(gm_fill) ** (1.0 / 3.0)
        quality_raw = roe_s * gm_s
        z_quality = _zscore(quality_raw)

        value_raw = ey_fill + cf_fill
        z_value = _zscore(value_raw)

        quality_gate = 0.5 + 0.5 * np.tanh(z_quality * QUALITY_S)
        value_gate = 0.5 + 0.5 * np.tanh(z_value * VALUE_S)

        score = (
            SIZE_W * size_sig * quality_gate * value_gate
            + QUALITY_W * z_quality
            + VALUE_W * z_value
        )

        return np.where(base_valid & np.isfinite(score), score, np.nan)
