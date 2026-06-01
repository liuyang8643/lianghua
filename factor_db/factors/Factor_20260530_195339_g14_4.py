import numpy as np

VALUE_W = 0.28
QUALITY_W = 0.22
GROWTH_W = 0.18
CF_W = 0.16
CASH_CONF_W = 0.10

CASH_CONF_S = 2.0
ROE_GATE_S = 2.0
GROWTH_GATE_S = 1.5

__thesis__ = "现金盈利确认的全因子质量门控"


class Factor_20260530_195339_g14_4:
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
            cash_conf = panel['operating_cf_ps'] / np.maximum(np.abs(panel['eps']), 1e-8)

        z_cash_conf = _zscore(cash_conf)
        cash_gate = 0.5 + 0.5 * np.tanh(z_cash_conf * CASH_CONF_S)

        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1/3))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1/3))
        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)
        z_quality_raw = _zscore(z_roe + z_gm)
        z_quality = _zscore(z_quality_raw * cash_gate)

        z_ey = _zscore(ey)
        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_GATE_S)
        z_value = _zscore(z_ey * roe_gate * cash_gate)

        g_profit = _zscore(panel['profit_yoy'])
        g_rev = _zscore(panel['revenue_yoy'])
        g_gate = 0.5 + 0.5 * np.tanh(g_profit * GROWTH_GATE_S)
        z_growth = _zscore((g_profit + g_rev) * g_gate)

        z_cf = _zscore(cf_yield)
        cf_sig = np.tanh(z_cf * CASH_CONF_S)


        score = (VALUE_W * z_value + QUALITY_W * z_quality + GROWTH_W * z_growth + CF_W * cf_sig + CASH_CONF_W * z_cash_conf)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
