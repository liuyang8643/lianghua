import numpy as np

VALUE_W = 0.40
QUALITY_W = 0.35
GATE_SHARP = 2.8

__thesis__ = "品质门控盈利市值比与小盘偏好融合"


class Factor_20260531_005210_g7_2:
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        base_valid = ~np.isnan(panel['open']) & (panel['open'] >= 2.0) & ~panel['st_mask']

        def _rank_norm(x):
            x = x.astype(np.float64)
            nan = np.isnan(x)
            order = np.argsort(np.argsort(np.where(nan, np.inf, x), axis=1), axis=1).astype(np.float64)
            n = (~nan).sum(axis=1, keepdims=True).astype(np.float64)
            r = 2.0 * order / np.maximum(n - 1.0, 1.0) - 1.0
            return np.where(nan, np.nan, r).astype(np.float32)

        def _zscore(x):
            x = x.astype(np.float64)
            mu = np.nanmean(x, axis=1, keepdims=True)
            sd = np.nanstd(x, axis=1, keepdims=True)
            return np.where(sd > 1e-12, (x - mu) / sd, np.nan).astype(np.float32)

        with np.errstate(divide='ignore', invalid='ignore'):
            ey = panel['eps'] / panel['open']
            cf_yield = panel['operating_cf_ps'] / panel['open']

        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1/3))
        z_roe = _zscore(roe_s)
        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * GATE_SHARP)

        z_ey = _zscore(ey)
        z_value = _zscore(z_ey * roe_gate)

        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1/3))
        z_gm = _zscore(gm_s)
        z_cf = _zscore(cf_yield)
        z_quality = _zscore(z_gm + z_cf)


        score = (VALUE_W * z_value + QUALITY_W * z_quality)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
