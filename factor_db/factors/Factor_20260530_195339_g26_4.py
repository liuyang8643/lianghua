import numpy as np

QUALITY_W = 0.28
VALUE_W = 0.20
GROWTH_QUAL_W = 0.16
CF_W = 0.14
INTERACT_W = 0.10

CASH_CONF_S = 2.0
ROE_GATE_S = 2.0
EY_CAP = 5.0

__thesis__ = "ROE与毛利率双现金门控的质量复合"


class Factor_20260530_195339_g26_4:
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        base_valid = ~np.isnan(panel['close']) & (panel['close'] >= 2.0) & ~panel['st_mask']

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
            ey = panel['eps'] / panel['close']
            cf_yield = panel['operating_cf_ps'] / panel['close']
            cash_coverage = panel['operating_cf_ps'] / np.where(np.isfinite(panel['eps']) & (np.abs(panel['eps']) > 1e-8), np.abs(panel['eps']), np.nan)

        z_cash = _zscore(cash_coverage)
        cash_gate = 0.5 + 0.5 * np.tanh(z_cash * CASH_CONF_S)

        roe_cb = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1/3))
        z_roe = _zscore(roe_cb)
        q_roe = z_roe * cash_gate

        gm_cb = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1/3))
        z_gm = _zscore(gm_cb)
        q_gm = z_gm * cash_gate

        z_quality = _rank_norm(q_roe + q_gm)

        ey_c = np.clip(ey, -EY_CAP, EY_CAP)
        z_ey = _zscore(ey_c)
        z_value = _zscore(z_ey * cash_gate)

        z_profit = _zscore(panel['profit_yoy'])
        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_GATE_S)
        z_growth = _zscore(z_profit * roe_gate)

        z_cf_yield = _zscore(cf_yield)
        cf_sig = np.tanh(z_cf_yield * CASH_CONF_S)

        z_interact = _rank_norm(q_roe * q_gm)


        score = (QUALITY_W * z_quality + VALUE_W * z_value + GROWTH_QUAL_W * z_growth + CF_W * cf_sig + INTERACT_W * z_interact)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
