import numpy as np

VALUE_W = 0.20
QUALITY_W = 0.18
GROWTH_W = 0.14
CF_W = 0.12
TRIANG_W = 0.14
CONV_W = 0.10
EFF_W = 0.08

CASH_S = 2.0
EARN_S = 2.0
EFF_S = 2.0
TRIANG_S = 2.0

__thesis__ = "三锚验证与利润营收收敛一致性因子"


class Factor_20260530_195339_g34_4:
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
            growth_eff = panel['profit_yoy'] / np.where(np.isfinite(panel['revenue_yoy']) & (np.abs(panel['revenue_yoy']) > 1e-8), np.abs(panel['revenue_yoy']), np.nan)

        # === Anchor 1: Cash (yield x coverage) ===
        z_cf_yield = _zscore(cf_yield)
        z_cash_cov = _zscore(cash_cov)
        cash_dual = _zscore(z_cf_yield + z_cash_cov)
        cash_anchor = 0.5 + 0.5 * np.tanh(cash_dual * CASH_S)

        # === Anchor 2: Earnings (ROE x margin integrity) ===
        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1/3))
        z_roe = _zscore(roe_s)
        earn_int = panel['gross_margin'] * np.tanh(panel['eps'] * 5.0)
        z_earn_int = _zscore(earn_int)
        earn_dual = _zscore(z_roe + z_earn_int)
        earn_anchor = 0.5 + 0.5 * np.tanh(earn_dual * EARN_S)

        # === Anchor 3: Efficiency (growth efficiency x margin) ===
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1/3))
        z_gm = _zscore(gm_s)
        z_growth_eff = _zscore(growth_eff)
        eff_anchor = 0.5 + 0.5 * np.tanh(z_growth_eff * EFF_S) * (0.5 + 0.5 * np.tanh(z_gm * EFF_S))

        # === Triangulation: three-anchor agreement ===
        r_cash = _rank_norm(cash_anchor)
        r_earn = _rank_norm(earn_anchor)
        r_eff = _rank_norm(eff_anchor)
        triang = 0.5 + 0.5 * np.tanh((r_cash + r_earn + r_eff) * TRIANG_S)

        # === Convergence: profit vs revenue rank alignment ===
        r_profit = _rank_norm(panel['profit_yoy'])
        r_revenue = _rank_norm(panel['revenue_yoy'])
        conv_qual = 1.0 - 0.5 * np.abs(r_profit - r_revenue)

        # === Component signals ===
        z_ey = _zscore(ey)
        z_value = _zscore(z_ey * earn_anchor * triang)

        z_quality = _zscore((z_roe + z_gm) * cash_anchor * triang)

        z_profit = _zscore(panel['profit_yoy'])
        z_revenue = _zscore(panel['revenue_yoy'])
        z_growth = _zscore((z_profit + z_revenue) * eff_anchor * conv_qual)

        cf_sig = np.tanh(z_cf_yield * CASH_S)
        eff_sig = np.tanh(z_growth_eff * EFF_S)
        z_conv = _zscore(conv_qual)


        score = (VALUE_W * z_value + QUALITY_W * z_quality + GROWTH_W * z_growth + CF_W * cf_sig + TRIANG_W * triang + CONV_W * z_conv + EFF_W * eff_sig)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
