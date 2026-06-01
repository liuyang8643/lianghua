import numpy as np

VALUE_W = 0.24
QUALITY_W = 0.20
SPIRAL_W = 0.16
ACCRUAL_W = 0.12
GROWTH_W = 0.12
LIFECYCLE_W = 0.08

CASH_S = 2.5
ROE_S = 2.0
SPIRAL_S = 1.8
ACCRUAL_S = 2.0
GROWTH_GATE_S = 1.5

__thesis__ = "效率锚定下的现金螺旋与应计共振复合体系"


class Factor_20260531_005210_g38_2:
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
            accrual_gap = (panel['operating_cf_ps'] - panel['eps']) / panel['open']
            ipo_ratio = panel['open'] / panel['issue_price']

        z_cash_conf = _zscore(cash_conf)
        cash_gate = 0.5 + 0.5 * np.tanh(z_cash_conf * CASH_S)

        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1.0 / 3.0))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1.0 / 3.0))
        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)

        z_ey = _zscore(ey)
        z_cf = _zscore(cf_yield)

        eff_raw = gm_s * cf_yield
        eff_s = np.sign(eff_raw) * (np.abs(eff_raw) ** (1.0 / 3.0))
        z_eff = _zscore(eff_s)
        eff_anchor = 0.5 + 0.5 * np.tanh(z_eff * CASH_S)

        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_S)
        value_raw = z_ey * roe_gate * cash_gate * eff_anchor
        z_value = _zscore(value_raw)

        z_quality = _zscore((z_roe + z_gm) * cash_gate)

        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])
        g_gate = 0.5 + 0.5 * np.tanh(z_profit * GROWTH_GATE_S)
        z_growth = _zscore((z_profit + z_rev) * g_gate * cash_gate)

        spiral_raw = np.tanh(z_cf * z_ey * z_eff * SPIRAL_S)
        z_spiral = _zscore(spiral_raw * cash_gate)

        r_ey = _rank_norm(z_ey)
        r_cf = _rank_norm(z_cf)
        resonance_agree = 1.0 - 0.5 * np.abs(r_ey - r_cf)
        resonance_dir = 0.5 + 0.5 * np.tanh((r_ey + r_cf) * ACCRUAL_S)
        z_accrual_gap = _zscore(accrual_gap)
        accrual_mod = 0.5 + 0.5 * np.tanh(z_accrual_gap * ACCRUAL_S)
        accrual_resonance = r_ey * r_cf * resonance_agree * resonance_dir * accrual_mod
        z_accrual = _zscore(np.tanh(accrual_resonance * ACCRUAL_S))

        z_lifecycle = _rank_norm(-np.log(np.maximum(ipo_ratio, 0.01)))


        score = (VALUE_W * z_value + QUALITY_W * z_quality + SPIRAL_W * z_spiral + ACCRUAL_W * z_accrual + GROWTH_W * z_growth + LIFECYCLE_W * z_lifecycle)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
