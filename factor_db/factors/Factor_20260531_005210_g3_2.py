import numpy as np

VALUE_W = 0.24
QUALITY_W = 0.20
GROWTH_W = 0.16
CF_W = 0.12
HEALTH_W = 0.10
INTERACT_W = 0.10

CF_GATE_S = 2.5
ROE_GATE_S = 1.8
HEALTH_S = 2.2
GROWTH_S = 1.6

__thesis__ = "现金双锚门控质量成长交叉共振"


class Factor_20260531_005210_g3_2:
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
            ey = panel['eps'] / panel['open']
            cf_yield = panel['operating_cf_ps'] / panel['open']
            cash_conf = panel['operating_cf_ps'] / np.where(np.isfinite(panel['eps']) & (np.abs(panel['eps']) > 1e-8), np.abs(panel['eps']), np.nan)

        z_cash_conf = _zscore(cash_conf)
        z_cf_yield = _zscore(cf_yield)
        z_cf_composite = _zscore(z_cash_conf + z_cf_yield)
        cash_gate = 0.5 + 0.5 * np.tanh(z_cf_composite * CF_GATE_S)

        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1/3))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1/3))
        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)
        z_quality = _rank_norm(_zscore(z_roe + z_gm) * cash_gate)

        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_GATE_S)
        z_ey = _zscore(ey)
        z_value = _zscore(z_ey * roe_gate * cash_gate)

        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])
        g_align = np.tanh(z_profit * z_rev * GROWTH_S)
        z_growth = _zscore(z_profit * z_rev * g_align)

        cf_sig = np.tanh(z_cf_yield * CF_GATE_S)

        health_raw = _zscore(z_cash_conf * z_gm)
        health_sig = np.tanh(health_raw * HEALTH_S)

        z_interact = _zscore(z_quality * z_ey)


        score = (VALUE_W * z_value + QUALITY_W * z_quality + GROWTH_W * z_growth + CF_W * cf_sig + HEALTH_W * health_sig + INTERACT_W * z_interact)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
