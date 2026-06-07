import numpy as np

QUALITY_W = 0.26
VALUE_W = 0.18
GROWTH_W = 0.14
COHERENCE_W = 0.12
ACCRUAL_W = 0.12
IPO_W = 0.10

CASH_SENS = 2.5
QUALITY_SENS = 2.0
GROWTH_SENS = 1.5
ACCRUAL_ASYM = 2.0

__thesis__ = "现金盈利一致性验证与不对称应计质量过滤"


class Factor_20260531_005210_g4_3:
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
            cash_cover = panel['operating_cf_ps'] / np.where(np.isfinite(panel['eps']) & (np.abs(panel['eps']) > 1e-8), np.abs(panel['eps']), np.nan)
            accruals = cf_yield - ey
            ipo_premium = panel['open'] / panel['issue_price']

        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1/3))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1/3))

        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)
        z_cash_cover = _zscore(cash_cover)
        z_cf = _zscore(cf_yield)
        z_ey = _zscore(ey)
        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])

        cash_gate = 0.5 + 0.5 * np.tanh(z_cash_cover * CASH_SENS)
        z_quality = _zscore((z_roe + z_gm) * cash_gate)

        quality_gate = 0.5 + 0.5 * np.tanh(z_quality * QUALITY_SENS)
        z_value = _zscore(z_ey * quality_gate * cash_gate)

        growth_raw = z_profit + z_rev
        g_gate = 0.5 + 0.5 * np.tanh(growth_raw * GROWTH_SENS)
        z_growth = _zscore(growth_raw * g_gate * quality_gate)

        coherence = 1.0 - np.abs(z_cf - z_ey) * 0.5
        z_coherence = _zscore(coherence)

        acr_penalty = np.tanh(np.maximum(accruals, 0.0) * ACCRUAL_ASYM)
        z_accrual = _zscore(z_cf - acr_penalty)

        log_ipo = np.where(np.isfinite(ipo_premium) & (ipo_premium > 0.01), -np.log(ipo_premium), np.nan)
        z_ipo = _zscore(log_ipo * quality_gate)


        score = (QUALITY_W * z_quality + VALUE_W * z_value + GROWTH_W * z_growth + COHERENCE_W * z_coherence + ACCRUAL_W * z_accrual + IPO_W * z_ipo)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
