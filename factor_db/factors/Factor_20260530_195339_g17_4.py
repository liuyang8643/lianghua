import numpy as np

VALUE_W = 0.22
QUALITY_W = 0.22
GROWTH_W = 0.18
CF_W = 0.12
PROFIT_CONF_W = 0.10
DIVER_W = 0.08
QUAL_INTERACT_W = 0.04

CASH_CONF_S = 2.0
ROE_GATE_S = 2.0
PROFIT_CONF_S = 3.0
EARN_STABILITY_S = 2.0
GROWTH_EFF_S = 1.5

__thesis__ = "利润现金双确认与增长效率门控"


class Factor_20260530_195339_g17_4:
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
            cash_conf = panel['operating_cf_ps'] / np.where(np.isfinite(panel['eps']) & (np.abs(panel['eps']) > 1e-8), np.abs(panel['eps']), np.nan)

        # Cash confirmation gate
        z_cash_conf = _zscore(cash_conf)
        cash_gate = 0.5 + 0.5 * np.tanh(z_cash_conf * CASH_CONF_S)

        # Profit consistency: eps × operating_cf directional alignment
        # Both positive or both negative = higher reliability
        eps_s = np.tanh(panel['eps'] * PROFIT_CONF_S)
        cf_s = np.tanh(panel['operating_cf_ps'] * 2.0)
        profit_consistency = eps_s * cf_s
        z_consistency = _zscore(profit_consistency)
        consistency_gate = 0.5 + 0.5 * np.tanh(z_consistency * EARN_STABILITY_S)

        # Quality: ROE + gross margin, dual-gated by cash + profit consistency
        roe_cb = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1/3))
        gm_cb = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1/3))
        z_roe = _zscore(roe_cb)
        z_gm = _zscore(gm_cb)
        z_quality_raw = _zscore(z_roe + z_gm)
        z_quality = _zscore(z_quality_raw * cash_gate * consistency_gate)

        # Value: earnings yield triple-gated by ROE + cash + consistency
        z_ey = _zscore(ey)
        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_GATE_S)
        z_value = _zscore(z_ey * roe_gate * cash_gate * consistency_gate)

        # Growth efficiency: profit growth per unit of revenue growth
        rev_safe = np.where(np.isfinite(panel['revenue_yoy']) & (np.abs(panel['revenue_yoy']) > 1e-8), np.abs(panel['revenue_yoy']), np.nan)
        growth_eff_raw = panel['profit_yoy'] / rev_safe
        growth_eff = np.tanh(growth_eff_raw * GROWTH_EFF_S)

        # Growth magnitude
        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])
        growth_mag = 0.5 * (z_profit + z_rev)

        # Combined growth: efficiency + magnitude, gated by consistency
        z_growth = _zscore((growth_eff + growth_mag) * consistency_gate)

        # Cash flow signal
        z_cf = _zscore(cf_yield)
        cf_sig = np.tanh(z_cf * CASH_CONF_S)

        # Revenue-profit diversion
        diversion = z_rev - z_profit
        z_diversion = _zscore(-diversion)

        # Quality-cash interaction
        z_interact = _zscore(z_quality * z_cash_conf)

        score = (VALUE_W * z_value + QUALITY_W * z_quality + GROWTH_W * z_growth + CF_W * cf_sig + PROFIT_CONF_W * z_consistency + DIVER_W * z_diversion + QUAL_INTERACT_W * z_interact)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
