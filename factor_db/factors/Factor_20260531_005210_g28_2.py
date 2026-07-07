import numpy as np

VALUE_W = 0.22
PRICING_POWER_W = 0.18
CASH_QUALITY_W = 0.16
GROWTH_W = 0.14
INTERACT_W = 0.10
LIFECYCLE_W = 0.08
CF_YIELD_W = 0.06

CASH_GATE_S = 2.5
ROE_GATE_S = 2.0
QUALITY_GATE_S = 2.0
POWER_S = 2.0
LIFE_S = 1.5

__thesis__ = "定价权传导：毛利确认现金质量驱动廉价价值"


class Factor_20260531_005210_g28_2:
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
            cash_conv = panel['operating_cf_ps'] / np.where(np.isfinite(panel['eps']) & (np.abs(panel['eps']) > 1e-8), np.abs(panel['eps']), np.nan)
            pricing_power = panel['profit_yoy'] - panel['revenue_yoy']

        gm_s = np.sign(panel['gross_margin']) * np.abs(panel['gross_margin']) ** (1/3)
        roe_s = np.sign(panel['roe']) * np.abs(panel['roe']) ** (1/3)

        z_gm = _zscore(gm_s)
        z_roe = _zscore(roe_s)
        z_cash_conv = _zscore(cash_conv)
        z_power = _zscore(pricing_power)
        z_ey = _zscore(ey)
        z_cf = _zscore(cf_yield)
        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])

        cash_gate = 0.5 + 0.5 * np.tanh(z_cash_conv * CASH_GATE_S)
        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_GATE_S)
        z_value = _zscore(z_ey * roe_gate * cash_gate)

        z_quality = _zscore(z_roe + z_gm)
        quality_gate = 0.5 + 0.5 * np.tanh(z_quality * QUALITY_GATE_S)
        z_growth = _zscore((z_profit + z_rev) * quality_gate)

        pricing_sig = np.tanh((z_gm + z_power) * POWER_S)

        cash_quality = _zscore(z_cash_conv * cash_gate)

        z_interact = _zscore(z_quality * z_ey)

        ipo_premium = panel['close'] / panel['issue_price']
        lifecycle = _rank_norm(np.where(np.isfinite(ipo_premium) & (ipo_premium > 0.01), -np.log(ipo_premium), np.nan))


        score = (VALUE_W * z_value + PRICING_POWER_W * pricing_sig + CASH_QUALITY_W * cash_quality + GROWTH_W * z_growth + INTERACT_W * z_interact + LIFECYCLE_W * lifecycle + CF_YIELD_W * z_cf)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
