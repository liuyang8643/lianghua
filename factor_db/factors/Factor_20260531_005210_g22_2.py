import numpy as np

QUALITY_MOM_W = 0.24
VALUE_ALIGN_W = 0.22
GROWTH_ACCRUAL_W = 0.18
IPO_QUAL_W = 0.12
CASH_DIRECT_W = 0.10

CASH_S = 2.5
ROE_S = 2.0
ALIGN_S = 2.0
ACCRUAL_S = 2.0
QUALITY_S = 2.0

__thesis__ = "质量动量加速与价值现金流对齐双重共振"


class Factor_20260531_005210_g22_2:
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

        z_cash = _zscore(cash_conf)
        cash_gate = 0.5 + 0.5 * np.tanh(z_cash * CASH_S)

        roe_cb = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1.0 / 3.0))
        gm_cb = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1.0 / 3.0))
        z_roe = _zscore(roe_cb)
        z_gm = _zscore(gm_cb)

        q_momentum = _zscore(z_roe * z_gm * cash_gate)
        z_quality = _zscore(q_momentum * QUALITY_S)

        z_ey = _zscore(ey)
        z_cf_yield = _zscore(cf_yield)
        alignment = 1.0 - np.abs(np.tanh(z_ey * ALIGN_S) - np.tanh(z_cf_yield * ALIGN_S))
        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_S)
        z_value = _zscore(z_ey * alignment * roe_gate * cash_gate)

        z_profit = _zscore(panel['profit_yoy'])
        accrual_gap = cf_yield - ey
        z_accrual = _zscore(accrual_gap)
        accrual_trust = 1.0 - np.abs(np.tanh(z_accrual * ACCRUAL_S))
        z_growth_accrual = _zscore(z_profit * accrual_trust * cash_gate)

        ipo_premium = panel['open'] / np.where(np.isfinite(panel['issue_price']) & (panel['issue_price'] > 1e-8), panel['issue_price'], np.nan)
        ipo_raw = _rank_norm(np.where(np.isfinite(ipo_premium) & (ipo_premium > 0.01), -np.log(ipo_premium), np.nan))
        z_ipo = _zscore(ipo_raw * cash_gate)

        cash_sig = np.tanh(z_cf_yield * CASH_S)


        score = (QUALITY_MOM_W * z_quality + VALUE_ALIGN_W * z_value + GROWTH_ACCRUAL_W * z_growth_accrual + IPO_QUAL_W * z_ipo + CASH_DIRECT_W * cash_sig)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
