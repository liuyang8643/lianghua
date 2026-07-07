import numpy as np

CASH_ANCHOR_W = 0.18
QUALITY_W = 0.16
VALUE_CHAIN_W = 0.16
RESONANCE_W = 0.14
GROWTH_W = 0.12
ACCRUAL_W = 0.10
OP_EFF_W = 0.08
LIFECYCLE_W = 0.04

CASH_S = 2.5
ROE_S = 2.0
GROWTH_S = 1.5
ACCRUAL_S = 2.0
OP_EFF_S = 2.0
RESONANCE_S = 2.0

__thesis__ = "现金锚定的质量杠杆共振链因子"


class Factor_20260531_005210_g17_4:
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
            cash_cover = panel['operating_cf_ps'] / np.where(np.isfinite(panel['eps']) & (np.abs(panel['eps']) > 1e-8), np.abs(panel['eps']), np.nan)
            accrual_gap = (panel['operating_cf_ps'] - panel['eps']) / panel['close']
            op_leverage = panel['gross_margin'] / np.where(np.isfinite(ey) & (np.abs(ey) > 1e-6), np.abs(ey), np.nan)
            ipo_premium = panel['close'] / np.where(np.isfinite(panel['issue_price']) & (panel['issue_price'] > 1e-8), panel['issue_price'], np.nan)

        roe_s = np.sign(panel['roe']) * np.abs(panel['roe']) ** (1.0 / 3.0)
        gm_s = np.sign(panel['gross_margin']) * np.abs(panel['gross_margin']) ** (1.0 / 3.0)

        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)
        z_ey = _zscore(ey)
        z_cf = _zscore(cf_yield)
        z_cover = _zscore(cash_cover)
        z_accrual = _zscore(accrual_gap)
        z_leverage = _zscore(op_leverage)
        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])

        z_cash_quality = _zscore(z_cover + z_cf)
        cash_gate = 0.5 + 0.5 * np.tanh(z_cash_quality * CASH_S)

        op_eff_sig = np.tanh(z_leverage * OP_EFF_S)

        quality_raw = z_roe * z_gm * cash_gate
        z_quality = _zscore(quality_raw * op_eff_sig)

        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_S)
        value_raw = z_ey * roe_gate * cash_gate
        z_value = _zscore(value_raw * op_eff_sig)

        profit_quality = np.tanh(z_profit * RESONANCE_S)
        cash_quality_sig = np.tanh(z_cash_quality * RESONANCE_S)
        resonance_raw = profit_quality * cash_quality_sig * (0.5 + 0.5 * z_roe * z_gm)
        z_resonance = _zscore(resonance_raw)

        growth_raw = z_profit * z_rev
        growth_gate = 0.5 + 0.5 * np.tanh((z_profit + z_rev) * GROWTH_S)
        z_growth = _zscore(growth_raw * growth_gate * cash_gate)

        accrual_sig = np.tanh(z_accrual * ACCRUAL_S)

        z_lifecycle = _rank_norm(np.where(np.isfinite(ipo_premium) & (ipo_premium > 0.01), -np.log(ipo_premium), np.nan))


        score = (CASH_ANCHOR_W * cash_quality_sig + QUALITY_W * z_quality + VALUE_CHAIN_W * z_value + RESONANCE_W * z_resonance + GROWTH_W * z_growth + ACCRUAL_W * accrual_sig + OP_EFF_W * op_eff_sig + LIFECYCLE_W * z_lifecycle)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
