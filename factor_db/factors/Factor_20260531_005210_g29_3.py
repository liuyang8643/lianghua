import numpy as np

VALUE_W = 0.20
QUALITY_W = 0.14
GROWTH_W = 0.12
VQ_RES_W = 0.18
QG_RES_W = 0.14
FRAGILITY_W = 0.10

CASH_S = 2.5
ROE_S = 2.0
GROWTH_EFF_S = 1.5
ACCRUAL_S = 2.0
RESONANCE_S = 2.0
FRAGILITY_S = 2.5

__thesis__ = "价值质量双共振网络剔除应计脆弱性"


class Factor_20260531_005210_g29_3:
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
            accruals = cf_yield - ey
            growth_eff = panel['profit_yoy'] / np.where(np.isfinite(panel['revenue_yoy']) & (np.abs(panel['revenue_yoy']) > 1e-8), np.abs(panel['revenue_yoy']), np.nan)

        z_cash_conf = _zscore(cash_conf)
        cash_gate = 0.5 + 0.5 * np.tanh(z_cash_conf * CASH_S)

        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1/3))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1/3))
        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)

        dual_gate = 0.5 + 0.5 * np.tanh((z_roe + z_cash_conf) * CASH_S)

        z_ey = _zscore(ey)
        z_cf = _zscore(cf_yield)
        z_accruals = _zscore(accruals)
        accrual_mod = 0.5 + 0.5 * np.tanh(z_accruals * ACCRUAL_S)

        z_value = _zscore(z_ey * accrual_mod * dual_gate)

        z_quality_raw = _zscore(z_roe + z_gm)
        z_quality = _zscore(z_quality_raw * cash_gate)

        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])
        g_gate = 0.5 + 0.5 * np.tanh(growth_eff * GROWTH_EFF_S)
        z_growth = _zscore((z_profit + z_rev) * g_gate * cash_gate)

        r_value = _rank_norm(z_value)
        r_quality = _rank_norm(z_quality_raw)
        r_growth = _rank_norm(z_growth)

        vq_agree = 1.0 - 0.5 * np.abs(r_value - r_quality)
        vq_direction = 0.5 + 0.5 * np.tanh((r_value + r_quality) * RESONANCE_S)
        vq_resonance = r_value * r_quality * vq_agree * vq_direction
        z_vq_res = _zscore(np.tanh(vq_resonance * RESONANCE_S))

        qg_agree = 1.0 - 0.5 * np.abs(r_quality - r_growth)
        qg_direction = 0.5 + 0.5 * np.tanh((r_quality + r_growth) * RESONANCE_S)
        qg_resonance = r_quality * r_growth * qg_agree * qg_direction
        z_qg_res = _zscore(np.tanh(qg_resonance * RESONANCE_S))

        fragility_raw = z_ey - z_cf
        z_fragility = _zscore(-np.tanh(fragility_raw * FRAGILITY_S))


        score = (VALUE_W * z_value + QUALITY_W * z_quality + GROWTH_W * z_growth + VQ_RES_W * z_vq_res + QG_RES_W * z_qg_res + FRAGILITY_W * z_fragility)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
