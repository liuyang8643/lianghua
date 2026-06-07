import numpy as np

VALUE_W = 0.22
QUALITY_W = 0.18
CASH_W = 0.16
GROWTH_W = 0.14
SPIRAL_W = 0.12
DIV_W = 0.10
FADE_W = 0.04

VALUE_S = 2.0
QUALITY_S = 3.0
CASH_S = 2.0
GROWTH_S = 1.8
SPIRAL_S = 1.5
DIV_S = 2.0
FADE_S = 2.0

__thesis__ = "现金穿透凹性价值与质量螺旋协同因子"


class Factor_20260531_005210_g2_0:
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        base_valid = ~np.isnan(panel['open']) & (panel['open'] >= 2.0) & ~panel['st_mask']

        def _rank_norm(x):
            x = x.astype(np.float64)
            nan = np.isnan(x)
            order = np.argsort(np.argsort(np.where(nan, np.inf, x), axis=1), axis=1).astype(np.float64)
            n = (~nan).sum(axis=1, keepdims=True).astype(np.float64)
            r = 2.0 * order / np.maximum(n - 1.0, 1.0) - 1.0
            return np.where(nan, np.nan, r).astype(np.float32)

        def _zscore(x):
            x = x.astype(np.float64)
            mu = np.nanmean(x, axis=1, keepdims=True)
            sd = np.nanstd(x, axis=1, keepdims=True)
            return np.where(sd > 1e-12, (x - mu) / sd, np.nan).astype(np.float32)

        with np.errstate(divide='ignore', invalid='ignore'):
            ey = panel['eps'] / panel['open']
            cf_yield = panel['operating_cf_ps'] / panel['open']
            cash_cov = panel['operating_cf_ps'] / np.where(np.isfinite(panel['eps']) & (np.abs(panel['eps']) > 1e-8), np.abs(panel['eps']), np.nan)
            ipo_premium = panel['open'] / np.where(np.isfinite(panel['issue_price']) & (panel['issue_price'] > 1e-8), panel['issue_price'], np.nan)

        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1.0 / 3.0))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1.0 / 3.0))
        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)

        z_ey = _zscore(ey)
        z_cf = _zscore(cf_yield)
        z_cov = _zscore(cash_cov)

        value_boost = 0.5 + 0.5 * np.tanh(np.abs(z_ey) * VALUE_S)
        z_value = _zscore(z_ey * value_boost)

        qual_raw = np.tanh(z_roe * z_gm * QUALITY_S)
        z_quality = _rank_norm(qual_raw)

        cash_raw = z_cf * (0.5 + 0.5 * np.tanh(z_cov * CASH_S))
        z_cash = _zscore(cash_raw)

        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])
        cf_gate = 0.5 + 0.5 * np.tanh(z_cf * GROWTH_S)
        z_growth = _rank_norm((z_profit + z_rev) * cf_gate)

        spiral_raw = np.tanh(z_ey * z_cf * z_roe * SPIRAL_S)
        z_spiral = _zscore(spiral_raw)

        div_raw = z_ey - z_cf
        z_div = _zscore(-np.tanh(div_raw * DIV_S))

        z_ipo = _rank_norm(np.where(np.isfinite(ipo_premium) & (ipo_premium > 0.01), -np.log(ipo_premium), np.nan))
        z_fade = _zscore(np.tanh(z_ipo * FADE_S))


        score = (VALUE_W * z_value + QUALITY_W * z_quality + CASH_W * z_cash + GROWTH_W * z_growth + SPIRAL_W * z_spiral + DIV_W * z_div + FADE_W * z_fade)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
