import numpy as np

VALUE_W = 0.20
QUALITY_W = 0.18
EFFICIENCY_W = 0.15
ACCRUAL_W = 0.13
RESONANCE_W = 0.12
GROWTH_W = 0.10
LIFECYCLE_W = 0.07

CASH_S = 2.5
ROE_S = 2.0
EFF_S = 1.8
ACCRUAL_S = 2.0
GROWTH_S = 1.5

__thesis__ = "现金门控价值质量共振叠加应计异象与效率锚"


class Factor_20260531_005210_g39_3:
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

        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1/3))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1/3))

        z_cash = _zscore(cash_conf)
        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)
        z_ey = _zscore(ey)
        z_cf = _zscore(cf_yield)
        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])
        z_accrual = _zscore(accrual_gap)

        cash_gate = 0.5 + 0.5 * np.tanh(z_cash * CASH_S)
        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_S)


        z_value = _zscore(z_ey * roe_gate * cash_gate)

        z_quality = _zscore((z_roe + z_gm) * cash_gate)

        eff_raw = gm_s * cf_yield
        eff_s = np.sign(eff_raw) * (np.abs(eff_raw) ** (1/3))
        z_eff = _zscore(eff_s)
        eff_sig = np.tanh(z_eff * EFF_S)

        accrual_sig = np.tanh(-z_accrual * ACCRUAL_S)

        r_ey = _rank_norm(ey)
        r_cf = _rank_norm(cf_yield)
        resonance_raw = r_ey * r_cf
        z_resonance = _zscore(resonance_raw * cash_gate)

        grow_interact = z_profit * z_rev
        grow_sig = np.tanh(grow_interact * GROWTH_S)
        z_growth = _zscore(grow_sig * cash_gate)

        z_lifecycle = _rank_norm(-np.log(np.maximum(ipo_ratio, 0.01)))


        score = (VALUE_W * z_value + QUALITY_W * z_quality + EFFICIENCY_W * eff_sig + ACCRUAL_W * accrual_sig + RESONANCE_W * z_resonance + GROWTH_W * z_growth + LIFECYCLE_W * z_lifecycle)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
