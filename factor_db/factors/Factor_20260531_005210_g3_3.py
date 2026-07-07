import numpy as np

VALUE_W = 0.20
QUALITY_W = 0.18
GROWTH_W = 0.16
HEALTH_W = 0.12
CF_W = 0.10
INTERACT_W = 0.09
DIVERGE_W = 0.07

CASH_S = 2.2
ROE_S = 2.0
GROWTH_S = 1.8
HEALTH_S = 2.0
RES_S = 1.6

__thesis__ = "三重门控共振与增长兑现交叉验证"


class Factor_20260531_005210_g3_3:
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
            cash_conf = panel['operating_cf_ps'] / np.where(np.isfinite(panel['eps']) & (np.abs(panel['eps']) > 1e-8), np.abs(panel['eps']), np.nan)

        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1/3))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1/3))
        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)
        z_ey = _zscore(ey)
        z_cf = _zscore(cf_yield)
        z_cash_conf = _zscore(cash_conf)
        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])

        cash_gate = 0.5 + 0.5 * np.tanh(z_cash_conf * CASH_S)
        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_S)

        profit_cf_align = np.tanh(panel['eps'] * panel['operating_cf_ps'] * 2.0)
        z_align = _zscore(profit_cf_align)
        align_gate = 0.5 + 0.5 * np.tanh(z_align * CASH_S)

        # Triple-gated value
        triple_gate = cash_gate * roe_gate * align_gate
        z_value = _zscore(z_ey * triple_gate)

        # ROE-CF resonance
        roe_cf_res = z_roe * z_cf
        z_roe_cf_res = _zscore(roe_cf_res)
        quality_resonance = np.tanh(z_roe_cf_res * RES_S)

        # Growth delivery
        rev_safe = np.where(np.isfinite(panel['revenue_yoy']) & (np.abs(panel['revenue_yoy']) > 1e-8), np.abs(panel['revenue_yoy']), np.nan)
        growth_eff = np.tanh(panel['profit_yoy'] / rev_safe * GROWTH_S)
        gm_sig = np.tanh(z_gm * 1.0)
        growth_delivery = growth_eff * gm_sig * (0.5 + 0.5 * z_profit)
        z_growth = _zscore(growth_delivery)

        # Health anchor (multiplicative)
        health_raw = z_cash_conf * z_gm
        z_health = _zscore(health_raw)
        health_sig = np.tanh(z_health * HEALTH_S)

        # CF direct
        cf_sig = np.tanh(z_cf * CASH_S)

        # Trust cross
        quality_raw = z_roe + z_gm
        trust_cross = quality_raw * cash_gate
        z_trust = _zscore(trust_cross)

        # Asymmetric profit bias
        divergence = z_profit - z_rev
        div_bias = np.where(divergence > 0, divergence * 1.5, divergence * 0.3)
        z_divergence = _zscore(div_bias)

        score = (VALUE_W * z_value + QUALITY_W * quality_resonance + GROWTH_W * z_growth + HEALTH_W * health_sig + CF_W * cf_sig + INTERACT_W * z_trust + DIVERGE_W * z_divergence)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
