import numpy as np

VALUE_W = 0.24
QUALITY_W = 0.20
GROWTH_ALIGN_W = 0.18
CF_W = 0.14
HEALTH_W = 0.10
INTERACT_W = 0.08

CASH_GATE_S = 2.0
ROE_GATE_S = 2.0
HEALTH_S = 2.0

__thesis__ = "现金确认质量与盈利趋势共振融合因子"


class Factor_20260530_195339_g28_4:
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

        # Cash confirmation gate (single coverage-based)
        z_cash = _zscore(cash_conf)
        cash_gate = 0.5 + 0.5 * np.tanh(z_cash * CASH_GATE_S)

        # Quality: tanh-transformed ROE + gross margin, cash-gated
        roe_t = np.tanh(panel['roe'] * 0.5)
        gm_t = np.tanh(panel['gross_margin'] * 2.0)
        z_quality = _zscore((_zscore(roe_t) + _zscore(gm_t)) * cash_gate)

        # Value: EY gated by ROE (profitability anchor) and cash
        z_roe = _zscore(roe_t)
        z_ey = _zscore(ey)
        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_GATE_S)
        z_value = _zscore(z_ey * roe_gate * cash_gate)

        # Growth alignment: profit and revenue must agree directionally
        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])
        z_growth = _zscore(z_profit * z_rev)

        # CF yield signal
        z_cf = _zscore(cf_yield)
        cf_sig = np.tanh(z_cf * CASH_GATE_S)

        # Financial health: cash coverage robustness
        health_sig = np.tanh(z_cash * HEALTH_S)

        # Interaction: quality x value
        z_interact = _zscore(z_quality * z_ey)

        score = (VALUE_W * z_value + QUALITY_W * z_quality + GROWTH_ALIGN_W * z_growth + CF_W * cf_sig + HEALTH_W * health_sig + INTERACT_W * z_interact)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
