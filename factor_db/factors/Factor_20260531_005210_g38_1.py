import numpy as np

VALUE_W = 0.20
QUALITY_W = 0.18
CONVEXITY_W = 0.16
SPIRAL_W = 0.14
PERSIST_W = 0.12
PURITY_W = 0.10
FADE_W = 0.06

CASH_S = 2.5
ROE_S = 2.0
SPIRAL_S = 2.0
CONVEX_S = 2.0
PERSIST_S = 1.5
PURITY_S = 2.5
FADE_S = 2.0

__thesis__ = "现金流凸性共振与多维度纯度螺旋衰减"


class Factor_20260531_005210_g38_1:
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
            ipo_premium = panel['open'] / np.where(np.isfinite(panel['issue_price']) & (panel['issue_price'] > 1e-8), panel['issue_price'], np.nan)

        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1/3))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1/3))

        z_ey = _zscore(ey)
        z_cf = _zscore(cf_yield)
        z_cash_conf = _zscore(cash_conf)
        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)

        cash_gate = 0.5 + 0.5 * np.tanh(z_cash_conf * CASH_S)
        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_S)

        z_value = _zscore(z_ey * cash_gate * roe_gate)

        z_quality = _zscore((z_roe + z_gm) * cash_gate)

        convex_raw = z_ey * z_cf * cash_gate
        z_convexity = _zscore(np.tanh(convex_raw * CONVEX_S))

        spiral_raw = gm_s * cf_yield * roe_gate
        z_spiral = _zscore(np.sign(spiral_raw) * np.abs(spiral_raw) ** (1/3))

        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])
        z_growth = _zscore((z_profit + z_rev) * cash_gate)

        r_v = _rank_norm(z_value)
        r_q = _rank_norm(z_quality)
        r_g = _rank_norm(z_growth)
        persist_raw = r_v * r_q * r_g
        z_persist = _zscore(np.tanh(persist_raw * PERSIST_S))

        accrual_div = z_ey - z_cf
        purity_raw = -np.tanh(accrual_div * PURITY_S) * cash_gate
        z_purity = _zscore(purity_raw)

        z_lifecycle = _rank_norm(np.where(np.isfinite(ipo_premium) & (ipo_premium > 0.01), -np.log(ipo_premium), np.nan))
        z_fade = _zscore(np.tanh(z_lifecycle * FADE_S))


        score = (VALUE_W * z_value + QUALITY_W * z_quality + CONVEXITY_W * z_convexity + SPIRAL_W * z_spiral + PERSIST_W * z_persist + PURITY_W * z_purity + FADE_W * z_fade)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
