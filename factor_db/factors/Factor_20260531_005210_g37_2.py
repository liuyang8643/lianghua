import numpy as np

STABILITY_W = 0.22
CERTAINTY_VAL_W = 0.20
QUAL_GROWTH_W = 0.18
CONVERGE_W = 0.16
FRAGILITY_W = 0.14

CASH_S = 2.5
ROE_S = 2.0
STAB_S = 1.8
VALUE_S = 2.0
GROWTH_S = 1.5
FRAGILITY_S = 2.5
CONVERGE_S = 1.5
ASYM_S = 0.5

__thesis__ = "利润现金确定性锚定的质价共振网络"


class Factor_20260531_005210_g37_2:
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

        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1.0 / 3.0))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1.0 / 3.0))

        z_cash = _zscore(cash_conf)
        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)
        z_ey = _zscore(ey)
        z_cf = _zscore(cf_yield)
        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])

        cash_gate = 0.5 + 0.5 * np.tanh(z_cash * CASH_S)
        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_S)
        dual_gate = cash_gate * roe_gate

        r_roe = _rank_norm(z_roe)
        r_gm = _rank_norm(z_gm)
        stability_raw = r_roe * r_gm * dual_gate
        z_stability = _zscore(np.tanh(stability_raw * STAB_S))

        cf_confirm = np.tanh((z_cf - z_ey) * VALUE_S)
        certainty_adj = 0.5 + 0.5 * cf_confirm
        cert_value_raw = z_ey * certainty_adj * dual_gate
        z_cert_value = _zscore(cert_value_raw)

        g_consistency = 1.0 - np.abs(np.tanh(z_profit * GROWTH_S) - np.tanh(z_rev * GROWTH_S))
        quality_gate = 0.5 + 0.5 * np.tanh((z_roe + z_gm) * STAB_S * 0.5)
        qg_raw = (z_profit + z_rev) * g_consistency * quality_gate * cash_gate
        z_qual_growth = _zscore(np.tanh(qg_raw * GROWTH_S))

        r_stability = _rank_norm(z_stability)
        r_cert = _rank_norm(z_cert_value)
        converge_sig = np.tanh(r_stability * r_cert * CONVERGE_S)
        z_converge = _zscore(converge_sig * dual_gate)

        fragility_raw = z_ey - z_cf
        asym_frag = np.where(fragility_raw < 0, fragility_raw * (1.0 + ASYM_S), fragility_raw * 0.5)
        z_fragility = _zscore(np.tanh(-asym_frag * FRAGILITY_S))


        score = (STABILITY_W * z_stability + CERTAINTY_VAL_W * z_cert_value + QUAL_GROWTH_W * z_qual_growth + CONVERGE_W * z_converge + FRAGILITY_W * z_fragility)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
