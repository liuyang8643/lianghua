import numpy as np

VQ_TRIPLE_W = 0.26
CASH_GROWTH_W = 0.22
CF_DIRECT_W = 0.14
GM_CF_EFF_W = 0.12
ACCRUAL_W = 0.10
IPO_W = 0.08

CASH_S = 2.0
ROE_S = 2.0
GM_S = 1.5
EY_S = 1.5
GROWTH_S = 1.5
ACCRUAL_S = 1.5

__thesis__ = "三维乘法共识与现金流增长交叉确认"


class Factor_20260531_005210_g15_4:
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
            ipo_premium = panel['open'] / panel['issue_price']

        z_roe = _zscore(np.sign(panel['roe']) * np.abs(panel['roe']) ** (1/3))
        z_gm = _zscore(np.sign(panel['gross_margin']) * np.abs(panel['gross_margin']) ** (1/3))
        z_ey = _zscore(ey)
        z_cf = _zscore(cf_yield)
        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])

        vq_triple = np.tanh(z_roe * ROE_S) * np.tanh(z_gm * GM_S) * np.tanh(z_ey * EY_S)

        cash_growth = np.tanh(z_cf * CASH_S) * np.tanh(z_profit * GROWTH_S) * np.tanh(z_rev * GROWTH_S)

        cf_direct = np.tanh(z_cf * CASH_S)

        gm_cf_eff = np.tanh(z_gm * GM_S) * np.tanh(z_cf * CASH_S)

        accrual_raw = z_ey - z_cf
        z_accrual = _zscore(accrual_raw)
        accrual = -np.tanh(z_accrual * ACCRUAL_S)

        z_ipo = _rank_norm(np.where(np.isfinite(ipo_premium) & (ipo_premium > 0.01), -np.log(ipo_premium), np.nan))

        score = (VQ_TRIPLE_W * vq_triple + CASH_GROWTH_W * cash_growth + CF_DIRECT_W * cf_direct + GM_CF_EFF_W * gm_cf_eff + ACCRUAL_W * accrual + IPO_W * z_ipo)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
