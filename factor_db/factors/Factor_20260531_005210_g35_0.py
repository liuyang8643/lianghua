import numpy as np

RESONANCE_W = 0.18
QUALITY_W = 0.16
CASH_W = 0.14
GROWTH_W = 0.12
ACCRUAL_W = 0.12
LEVERAGE_W = 0.10
IPO_W = 0.10

CASH_S = 2.5
QUALITY_S = 2.0
RESONANCE_S = 2.0
GROWTH_S = 1.5
ACCRUAL_S = 2.0
LEVERAGE_S = 1.8

__thesis__ = "双收益共振的品质桥接成长应计因子"


class Factor_20260531_005210_g35_0:
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
            accrual_gap = (panel['operating_cf_ps'] - panel['eps']) / panel['open']
            op_leverage = panel['gross_margin'] / np.where(np.isfinite(ey) & (np.abs(ey) > 1e-6), np.abs(ey), np.nan)
            ipo_premium = panel['open'] / np.where(np.isfinite(panel['issue_price']) & (panel['issue_price'] > 1e-8), panel['issue_price'], np.nan)

        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1.0 / 3.0))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1.0 / 3.0))

        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)
        z_ey = _zscore(ey)
        z_cf = _zscore(cf_yield)
        z_cash = _zscore(cash_conf)
        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])
        z_accrual = _zscore(accrual_gap)
        z_leverage = _zscore(op_leverage)

        cash_gate = 0.5 + 0.5 * np.tanh(z_cash * CASH_S)
        quality_gate = 0.5 + 0.5 * np.tanh((z_roe + z_gm) * QUALITY_S)

        dual_energy = np.sqrt(np.where(np.isfinite(z_ey ** 2 + z_cf ** 2) & (z_ey ** 2 + z_cf ** 2 > 1e-12), z_ey ** 2 + z_cf ** 2, np.nan))
        dual_alignment = z_ey * z_cf / (1.0 + np.abs(z_ey - z_cf))
        resonance = _zscore(dual_alignment * dual_energy * cash_gate * RESONANCE_S)

        quality_geo = np.sign(roe_s * gm_s) * np.sqrt(np.abs(roe_s * gm_s))
        z_quality_geo = _zscore(quality_geo)
        bridge = _zscore(z_quality_geo * z_cash * quality_gate * cash_gate)

        cash_score = _zscore(z_cf + z_cash) * cash_gate

        r_profit = _rank_norm(panel['profit_yoy'])
        r_rev = _rank_norm(panel['revenue_yoy'])
        conv = 1.0 - 0.5 * np.abs(r_profit - r_rev)
        growth_raw = z_profit + z_rev
        z_growth = _zscore(growth_raw * conv * np.tanh(z_gm * GROWTH_S) * cash_gate)

        accrual_score = np.tanh(z_accrual * ACCRUAL_S) * cash_gate

        leverage_score = np.tanh(z_leverage * LEVERAGE_S)

        z_ipo = _rank_norm(np.where(np.isfinite(ipo_premium) & (ipo_premium > 0.01), -np.log(ipo_premium), np.nan))


        score = (RESONANCE_W * resonance + QUALITY_W * bridge + CASH_W * cash_score + GROWTH_W * z_growth + ACCRUAL_W * accrual_score + LEVERAGE_W * leverage_score + IPO_W * z_ipo)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
