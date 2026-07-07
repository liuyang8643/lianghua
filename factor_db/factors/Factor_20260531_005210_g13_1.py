import numpy as np

VALUE_W = 0.22
PROFIT_SPIRAL_W = 0.20
GROWTH_QUALITY_W = 0.16
ACCRUAL_CONF_W = 0.14
EFFICIENCY_W = 0.12
IPO_W = 0.08

CASH_S = 2.0
ROE_S = 2.0
GM_S = 2.0
GROWTH_DIV_S = 1.5
CONFIRM_S = 2.5

__thesis__ = "现金流确认的盈利螺旋与增长质量分化融合打分"


class Factor_20260531_005210_g13_1:
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
            cash_cov = panel['operating_cf_ps'] / np.where(np.isfinite(panel['eps']) & (np.abs(panel['eps']) > 1e-8), np.abs(panel['eps']), np.nan)
            accruals = cf_yield - ey
            growth_div = panel['profit_yoy'] - panel['revenue_yoy']
            ipo_premium = panel['close'] / np.where(np.isfinite(panel['issue_price']) & (panel['issue_price'] > 1e-8), panel['issue_price'], np.nan)

        z_cf = _zscore(cf_yield)
        z_cov = _zscore(cash_cov)
        z_cash_quality = _zscore(z_cf + z_cov)
        cash_gate = 0.5 + 0.5 * np.tanh(z_cash_quality * CASH_S)

        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1.0 / 3.0))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1.0 / 3.0))
        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)

        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_S)
        gm_gate = 0.5 + 0.5 * np.tanh(z_gm * GM_S)

        confirm1 = z_roe * cash_gate
        confirm2 = _zscore(confirm1) * gm_gate
        z_spiral = _zscore(confirm2)

        z_ey = _zscore(ey)
        z_value = _zscore(z_ey * roe_gate * cash_gate * z_spiral)

        confirm_gate = 0.5 + 0.5 * np.tanh(z_spiral * CONFIRM_S)
        z_accruals = _zscore(accruals)
        z_accrual_conf = _zscore(z_accruals * confirm_gate)

        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])
        z_div = _zscore(growth_div)

        both_pos = (0.5 + 0.5 * np.tanh(z_profit * GROWTH_DIV_S)) * (0.5 + 0.5 * np.tanh(z_rev * GROWTH_DIV_S))
        z_growth_quality = _zscore(z_div * both_pos * cash_gate)

        z_eff = _zscore(panel['profit_yoy'] / np.where(np.isfinite(panel['revenue_yoy']) & (np.abs(panel['revenue_yoy']) > 1e-8), np.abs(panel['revenue_yoy']), np.nan))
        z_efficiency = _zscore(z_eff * gm_gate)

        z_ipo = _rank_norm(np.where(np.isfinite(ipo_premium) & (ipo_premium > 0.01), -np.log(ipo_premium), np.nan))


        score = (VALUE_W * z_value + PROFIT_SPIRAL_W * z_spiral + GROWTH_QUALITY_W * z_growth_quality + ACCRUAL_CONF_W * z_accrual_conf + EFFICIENCY_W * z_efficiency + IPO_W * z_ipo)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
