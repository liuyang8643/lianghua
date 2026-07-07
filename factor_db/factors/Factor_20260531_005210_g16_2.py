import numpy as np

CHAIN_VALUE_W = 0.28
CASH_QUALITY_W = 0.16
EARN_QUALITY_W = 0.14
GROWTH_COHERENCE_W = 0.14
ACCRUAL_W = 0.10
MARGIN_LEVEL_W = 0.08
LIFECYCLE_W = 0.05

CF_S = 2.0
ROE_S = 2.5
GM_S = 1.5
GROWTH_S = 1.5
ACCRUAL_S = 2.0
CHAIN_S = 2.0

__thesis__ = "质量传导链逐级验证：现金确认盈利驱动价值"


class Factor_20260531_005210_g16_2:
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
            cash_coverage = panel['operating_cf_ps'] / np.where(np.isfinite(panel['eps']) & (np.abs(panel['eps']) > 1e-8), np.abs(panel['eps']), np.nan)
            ipo_premium = panel['close'] / panel['issue_price']

        roe_s = np.sign(panel['roe']) * np.abs(panel['roe']) ** (1/3)
        gm_s = np.sign(panel['gross_margin']) * np.abs(panel['gross_margin']) ** (1/3)

        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)
        z_ey = _zscore(ey)
        z_cf = _zscore(cf_yield)
        z_coverage = _zscore(cash_coverage)
        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])

        cash_quality = _zscore(z_cf + z_coverage)
        cash_sig = np.tanh(cash_quality * CF_S)

        earn_quality = _zscore(z_roe + z_gm)
        earn_sig = np.tanh(earn_quality * ROE_S)

        cf_gate = 0.5 + 0.5 * np.tanh(cash_quality * CHAIN_S)
        earn_confirmed = np.tanh(earn_quality * cf_gate * ROE_S)
        earn_gate = 0.5 + 0.5 * earn_confirmed
        chain_value = _zscore(z_ey * earn_gate)

        rev_gate = 0.5 + 0.5 * np.tanh(z_rev * GROWTH_S)
        growth_coherence = _zscore(z_profit * rev_gate)

        accrual_gap = z_ey - z_cf
        accrual = -np.tanh(accrual_gap * ACCRUAL_S)

        margin_sig = np.tanh(z_gm * GM_S)

        lifecycle = _rank_norm(np.where(np.isfinite(ipo_premium) & (ipo_premium > 0.01), -np.log(ipo_premium), np.nan))


        score = (CHAIN_VALUE_W * chain_value + CASH_QUALITY_W * cash_sig + EARN_QUALITY_W * earn_sig + GROWTH_COHERENCE_W * growth_coherence + ACCRUAL_W * accrual + MARGIN_LEVEL_W * margin_sig + LIFECYCLE_W * lifecycle)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
