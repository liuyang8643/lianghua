import numpy as np

CASH_QUAL_W = 0.28
GROWTH_STAB_W = 0.24
VALUE_ALT_W = 0.20
MARGIN_POWER_W = 0.16

CASH_QUAL_S = 2.5
GROWTH_STAB_S = 2.0
VALUE_S = 2.0
MARGIN_S = 2.0

__thesis__ = "现金盈利背离与质量增长双螺旋选股"


class Factor_20260531_005210_g29_2:
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
            cash_to_earn = cf_yield / np.where(np.isfinite(ey) & (np.abs(ey) > 1e-8), np.abs(ey), np.nan)
            growth_stability = panel['profit_yoy'] / np.where(np.isfinite(panel['revenue_yoy']) & (np.abs(panel['revenue_yoy']) > 1e-8), np.abs(panel['revenue_yoy']), np.nan)

        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1/3))
        z_gm = _zscore(gm_s)
        gm_gate = 0.5 + 0.5 * np.tanh(z_gm * MARGIN_S)

        z_cf = _zscore(cf_yield)
        z_cash_to_earn = _zscore(cash_to_earn)
        cash_qual = _rank_norm(z_cash_to_earn * gm_gate * np.tanh(z_cf * CASH_QUAL_S))

        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])
        z_stab = _zscore(growth_stability)
        growth_stab = _zscore((z_profit + z_rev) * np.tanh(z_stab * GROWTH_STAB_S))

        z_ey = _zscore(ey)
        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1/3))
        z_roe = _zscore(roe_s)
        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * VALUE_S)
        value_alt = _zscore((z_ey + z_cf) * roe_gate)

        margin_power = _rank_norm(z_gm * gm_gate)


        score = (CASH_QUAL_W * cash_qual + GROWTH_STAB_W * growth_stab + VALUE_ALT_W * value_alt + MARGIN_POWER_W * margin_power)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
