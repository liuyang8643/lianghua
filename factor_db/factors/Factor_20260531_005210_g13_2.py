import numpy as np

VALUE_W = 0.22
CASCADE_W = 0.18
GROWTH_W = 0.16
CF_W = 0.12
CONV_W = 0.10
ACC_W = 0.08
LEV_W = 0.06
IPO_W = 0.04

CASCADE_S = 2.5
CF_S = 2.0
LEV_S = 1.5
ACC_S = 2.0

__thesis__ = "三层品质级联与经营杠杆放大融合因子"


class Factor_20260531_005210_g13_2:
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
            accruals = (panel['operating_cf_ps'] - panel['eps']) / panel['close']
            op_leverage = panel['gross_margin'] / np.where(np.isfinite(ey) & (np.abs(ey) > 1e-6), np.abs(ey), np.nan)
            ipo_premium = panel['close'] / np.where(np.isfinite(panel['issue_price']) & (panel['issue_price'] > 1e-8), panel['issue_price'], np.nan)

        # Layer 1: Cash foundation — yield + coverage fused
        z_cf_yield = _zscore(cf_yield)
        z_cash_cover = _zscore(cash_cover)
        cash_quality = _zscore(z_cf_yield + z_cash_cover)
        cash_gate = 0.5 + 0.5 * np.tanh(cash_quality * CF_S)

        # Layer 2: Earnings integrity — ROE + GM validated by cash gate
        roe_s = np.sign(panel['roe']) * np.abs(panel['roe']) ** (1.0 / 3.0)
        gm_s = np.sign(panel['gross_margin']) * np.abs(panel['gross_margin']) ** (1.0 / 3.0)
        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)
        earn_raw = _zscore(z_roe + z_gm)
        z_earn = _zscore(earn_raw * cash_gate)
        earn_gate = 0.5 + 0.5 * np.tanh(z_earn * CASCADE_S)

        # Layer 3: Growth conviction — profit×revenue gated by earnings
        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])
        growth_conv = np.sign(z_profit) * np.sqrt(np.abs(z_profit * z_rev))
        z_growth = _zscore(growth_conv * earn_gate)

        # Operating leverage: GM per unit EY — amplifier for value
        z_lev = _zscore(op_leverage)
        lev_sig = np.tanh(z_lev * LEV_S)

        # Value: EY amplified by earnings gate × leverage
        z_ey = _zscore(ey)
        z_value = _zscore(z_ey * earn_gate * (0.5 + 0.5 * lev_sig))

        # Cascade: sequential quality chain
        z_cascade = _zscore(cash_gate + earn_gate)

        # Accrual penalty: magnitude in either direction penalizes
        z_accruals = _zscore(accruals)
        accrual_penalty = -np.tanh(np.abs(z_accruals) * ACC_S)

        # Profit-revenue convergence
        r_profit = _rank_norm(panel['profit_yoy'])
        r_revenue = _rank_norm(panel['revenue_yoy'])
        convergence = 1.0 - 0.5 * np.abs(r_profit - r_revenue)
        z_conv = _zscore(convergence)

        # Direct cash signal
        cf_sig = np.tanh(z_cf_yield * CF_S)

        z_ipo = _rank_norm(np.where(np.isfinite(ipo_premium) & (ipo_premium > 0.01), -np.log(ipo_premium), np.nan))

        score = (VALUE_W * z_value + CASCADE_W * z_cascade + GROWTH_W * z_growth + CF_W * cf_sig + CONV_W * z_conv + ACC_W * accrual_penalty + LEV_W * lev_sig + IPO_W * z_ipo)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
