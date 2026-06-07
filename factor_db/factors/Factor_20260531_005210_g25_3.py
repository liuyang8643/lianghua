import numpy as np

VALUE_W = 0.26
QUALITY_W = 0.22
GROWTH_W = 0.18
CF_W = 0.14
EQ_SPREAD_W = 0.12

__thesis__ = "乘积质量门控与盈利现金价差交叉验证"


class Factor_20260531_005210_g25_3:
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

        z_ey = _zscore(ey)
        z_cf = _zscore(cf_yield)
        z_cash = _zscore(cash_conf)

        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1.0 / 3.0))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1.0 / 3.0))
        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)
        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])


        cash_gate = 0.5 + 0.5 * np.tanh(z_cash * 3.0)
        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * 2.0)

        z_value = _zscore(z_ey * roe_gate * cash_gate)

        z_quality = _zscore(z_roe * z_gm * cash_gate)

        growth_agree = np.tanh(z_profit * z_rev * 2.0)
        z_growth = _zscore((z_profit + z_rev) * growth_agree * cash_gate)

        z_cf_final = _zscore(z_cf * cash_gate)

        eq_spread = _zscore(z_cf - z_ey)


        score = (VALUE_W * z_value + QUALITY_W * z_quality + GROWTH_W * z_growth + CF_W * z_cf_final + EQ_SPREAD_W * eq_spread)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
