import numpy as np

VALUE_W = 0.26
QUALITY_W = 0.22
GROWTH_W = 0.18
CF_W = 0.14
DIVER_W = 0.08
IPO_W = 0.06

CASH_CONF_S = 2.0
ROE_GATE_S = 2.0
GROWTH_GATE_S = 1.5
DIVER_CAP = 3.0

__thesis__ = "现金流确认的质量增长双门控与折价增强"


class Factor_20260530_195339_g21_4:
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        base_valid = ~np.isnan(panel['close']) & (panel['close'] >= 2.0) & ~panel['st_mask']

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
            ey = panel['eps'] / panel['close']
            cf_yield = panel['operating_cf_ps'] / panel['close']
            cash_coverage = panel['operating_cf_ps'] / np.where(np.isfinite(panel['eps']) & (np.abs(panel['eps']) > 1e-8), np.abs(panel['eps']), np.nan)
            ipo_premium = panel['close'] / panel['issue_price']

        # Cash dual confirmation gate
        z_cf_yield = _zscore(cf_yield)
        z_coverage = _zscore(cash_coverage)
        z_cash = _zscore(z_cf_yield + z_coverage)
        cash_gate = 0.5 + 0.5 * np.tanh(z_cash * CASH_CONF_S)
        cf_sig = np.tanh(z_cf_yield * CASH_CONF_S)

        # Quality: cube-root ROE + GM, cash-confirmed
        roe_cb = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1/3))
        gm_cb = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1/3))
        z_roe = _zscore(roe_cb)
        z_gm = _zscore(gm_cb)
        z_quality_raw = _zscore(z_roe + z_gm)
        z_quality = _zscore(z_quality_raw * cash_gate)

        # Value: earnings yield gated by ROE + cash confirmation
        z_ey = _zscore(ey)
        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_GATE_S)
        z_value = _zscore(z_ey * roe_gate * cash_gate)

        # Growth: profit + revenue growth gated by profit momentum
        g_profit = _zscore(panel['profit_yoy'])
        g_rev = _zscore(panel['revenue_yoy'])
        g_gate = 0.5 + 0.5 * np.tanh(g_profit * GROWTH_GATE_S)
        z_growth = _zscore((g_profit + g_rev) * g_gate)

        # Revenue-profit divergence: revenue without profit is low quality
        diversion = np.clip(g_rev - g_profit, -DIVER_CAP, DIVER_CAP)
        z_diversion = _zscore(diversion)

        # IPO discount (z-score instead of parent's rank_norm)
        z_ipo = _zscore(np.where(np.isfinite(ipo_premium) & (ipo_premium > 0.01), -np.log(ipo_premium), np.nan))


        score = (VALUE_W * z_value + QUALITY_W * z_quality + GROWTH_W * z_growth + CF_W * cf_sig + DIVER_W * z_diversion + IPO_W * z_ipo)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
