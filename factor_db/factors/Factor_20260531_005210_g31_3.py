import numpy as np

VALUE_W = 0.22
QUALITY_W = 0.20
MOMENTUM_W = 0.14
EFFICIENCY_W = 0.12
DISPERSION_W = 0.10
HEALTH_W = 0.08
COHERENCE_W = 0.08

CASH_GATE_S = 2.5
ROE_GATE_S = 2.0
MOM_S = 1.8
HEALTH_S = 2.0
EFF_S = 1.5

__thesis__ = "现金锚定价值质量动量三维分歧惩罚增强"


class Factor_20260531_005210_g31_3:
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
            cash_conf = panel['operating_cf_ps'] / np.maximum(np.abs(panel['eps']), 1e-8)

        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1.0 / 3.0))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1.0 / 3.0))

        z_cash_conf = _zscore(cash_conf)
        cash_gate = 0.5 + 0.5 * np.tanh(z_cash_conf * CASH_GATE_S)

        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)
        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_GATE_S)

        z_ey = _zscore(ey)
        value_raw = z_ey * roe_gate * cash_gate
        z_value = _zscore(value_raw)

        quality_raw = (z_roe + z_gm) * cash_gate
        z_quality = _zscore(quality_raw)

        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])
        cash_conf_s = np.sign(cash_conf) * (np.abs(cash_conf) ** (1.0 / 3.0))
        z_cash_conf_s = _zscore(cash_conf_s)
        mom_gate = 0.5 + 0.5 * np.tanh(z_cash_conf_s * MOM_S)
        momentum_raw = (z_profit + z_rev) * mom_gate
        z_momentum = _zscore(momentum_raw)

        cf_sig = np.tanh(cf_yield * EFF_S)
        efficiency_raw = gm_s * cf_sig
        z_efficiency = _zscore(efficiency_raw)

        r_val = _rank_norm(value_raw)
        r_qual = _rank_norm(quality_raw)
        r_mom = _rank_norm(momentum_raw)
        mean_r = (r_val + r_qual + r_mom) / 3.0
        dispersion = np.sqrt(((r_val - mean_r)**2 + (r_qual - mean_r)**2 + (r_mom - mean_r)**2) / 3.0)
        z_dispersion = _zscore(-dispersion)

        cf_yield_s = np.sign(cf_yield) * (np.abs(cf_yield) ** (1.0 / 3.0))
        z_cf_s = _zscore(cf_yield_s)
        health_raw = z_cash_conf * (0.5 + 0.5 * np.tanh(z_cf_s * HEALTH_S))
        z_health = _zscore(health_raw)

        coherence_raw = value_raw * quality_raw
        z_coherence = _zscore(coherence_raw)


        score = (VALUE_W * z_value + QUALITY_W * z_quality + MOMENTUM_W * z_momentum + EFFICIENCY_W * z_efficiency + DISPERSION_W * z_dispersion + HEALTH_W * z_health + COHERENCE_W * z_coherence)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
