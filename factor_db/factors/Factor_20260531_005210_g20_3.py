import numpy as np

CONSENSUS_W = 0.22
VALUE_W = 0.18
CROSS_W = 0.14
GROWTH_W = 0.14
ACCRUAL_W = 0.12
IPO_W = 0.10

CASH_S = 2.5
ROE_S = 2.0
GM_S = 2.0
EY_S = 2.0
EFF_S = 2.5
GROWTH_S = 2.0
ACCRUAL_S = 2.5

__thesis__ = "三对共识效率桥与主门控增长应计交叉融合"


class Factor_20260531_005210_g20_3:
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
            ipo_premium = panel['open'] / panel['issue_price']
            accruals = cf_yield - ey

        roe_cb = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1/3))
        gm_cb = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1/3))
        z_roe = _zscore(roe_cb)
        z_gm = _zscore(gm_cb)
        z_ey = _zscore(ey)
        z_cf = _zscore(cf_yield)
        z_cash_conf = _zscore(cash_conf)
        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])

        cash_gate = 0.5 + 0.5 * np.tanh(z_cash_conf * CASH_S)
        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_S)
        master_gate = cash_gate * roe_gate

        t_roe = np.tanh(z_roe * ROE_S)
        t_gm = np.tanh(z_gm * GM_S)
        t_ey = np.tanh(z_ey * EY_S)
        pairwise_sum = t_roe * t_gm + t_roe * t_ey + t_gm * t_ey
        z_consensus = _zscore(pairwise_sum * master_gate)

        eff_bridge = np.tanh(z_gm * z_cf * EFF_S)
        z_value = _zscore(z_ey * eff_bridge * master_gate)

        cross_quality = (z_roe + z_gm) * (z_cf + z_cash_conf)
        z_cross = _zscore(cross_quality * cash_gate)

        growth_conf = z_profit * z_rev
        z_growth = _zscore(np.tanh(growth_conf * GROWTH_S) * cash_gate)

        z_accruals = _zscore(accruals)
        asym_accrual = np.where(z_accruals < 0, z_accruals * 2.0, z_accruals * 0.5)
        z_accrual = _zscore(np.tanh(asym_accrual * ACCRUAL_S))

        z_ipo = _rank_norm(np.where(np.isfinite(ipo_premium) & (ipo_premium > 0.01), -np.log(ipo_premium), np.nan))


        score = (CONSENSUS_W * z_consensus + VALUE_W * z_value + CROSS_W * z_cross + GROWTH_W * z_growth + ACCRUAL_W * z_accrual + IPO_W * z_ipo)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
