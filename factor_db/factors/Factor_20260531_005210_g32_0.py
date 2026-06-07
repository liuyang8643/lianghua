import numpy as np

VALUE_W = 0.22
QUALITY_W = 0.16
GROWTH_W = 0.14
TRUST_W = 0.14
RESONANCE_W = 0.12
CONV_W = 0.10
IPO_W = 0.06

CASH_S = 2.5
EARN_S = 2.0
ACCRUAL_S = 2.0
TRIANG_S = 2.0
TRUST_S = 1.8
RESONANCE_S = 2.0
GROWTH_S = 1.8
CONV_S = 2.0

__thesis__ = "三锚信任三角应计共振成长收敛融合因子"


class Factor_20260531_005210_g32_0:
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
            cash_cov = panel['operating_cf_ps'] / np.where(np.isfinite(panel['eps']) & (np.abs(panel['eps']) > 1e-8), np.abs(panel['eps']), np.nan)
            accrual_gap = (panel['operating_cf_ps'] - panel['eps']) / panel['open']
            op_leverage = panel['gross_margin'] / np.where(np.isfinite(ey) & (np.abs(ey) > 1e-6), np.abs(ey), np.nan)
            ipo_premium = panel['open'] / np.where(np.isfinite(panel['issue_price']) & (panel['issue_price'] > 1e-8), panel['issue_price'], np.nan)

        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1.0 / 3.0))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1.0 / 3.0))

        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)
        z_cf = _zscore(cf_yield)
        z_cash_cov = _zscore(cash_cov)
        z_accrual = _zscore(accrual_gap)
        z_leverage = _zscore(op_leverage)
        z_ey = _zscore(ey)
        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])

        # Anchor 1: Cash (yield + coverage dual)
        cash_dual = _zscore(z_cf + z_cash_cov)
        cash_anchor = 0.5 + 0.5 * np.tanh(cash_dual * CASH_S)

        # Anchor 2: Earnings quality (ROE x margin integrity)
        earn_int = panel['gross_margin'] * np.tanh(panel['eps'] * 5.0)
        z_earn_int = _zscore(earn_int)
        earn_dual = _zscore(z_roe + z_earn_int)
        earn_anchor = 0.5 + 0.5 * np.tanh(earn_dual * EARN_S)

        # Anchor 3: Accrual quality (accrual gap x cash gate)
        cash_gate = 0.5 + 0.5 * np.tanh(z_cash_cov * CASH_S)
        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * EARN_S)
        accrual_anchor = 0.5 + 0.5 * np.tanh(z_accrual * ACCRUAL_S) * cash_gate

        # Triangulation: three-anchor agreement
        r_cash = _rank_norm(cash_anchor)
        r_earn = _rank_norm(earn_anchor)
        r_accrual = _rank_norm(accrual_anchor)
        triang = 0.5 + 0.5 * np.tanh((r_cash + r_earn + r_accrual) * TRIANG_S)

        # Trust bridge: cash-quality x earnings-quality x triangulation
        cash_quality = z_cf + z_cash_cov
        earn_quality = z_roe + z_gm
        trust_raw = cash_quality * earn_quality
        z_trust = _zscore(np.tanh(trust_raw * TRUST_S) * triang)

        # Accrual x operating leverage resonance
        resonance_raw = np.tanh(z_accrual * ACCRUAL_S) * np.tanh(z_leverage * RESONANCE_S / 2.0)
        z_resonance = _zscore(resonance_raw)

        # Convergence: profit vs revenue rank alignment
        r_profit = _rank_norm(panel['profit_yoy'])
        r_rev = _rank_norm(panel['revenue_yoy'])
        conv_raw = 1.0 - 0.5 * np.abs(r_profit - r_rev)
        z_conv = _zscore(np.tanh(conv_raw * CONV_S))

        # Growth: dual engine with triangulation
        z_growth = _zscore((z_profit + z_rev) * conv_raw * triang)

        # Value: EY x multi-gate
        z_value = _zscore(z_ey * roe_gate * cash_gate * triang)

        # Quality: ROE + GM x cash & accrual anchors
        z_quality = _zscore((z_roe + z_gm) * cash_anchor * accrual_anchor)

        # IPO fade: favor stocks nearer to issue price
        z_ipo = _rank_norm(np.where(np.isfinite(ipo_premium) & (ipo_premium > 0.01), -np.log(ipo_premium), np.nan))

        score = (VALUE_W * z_value + QUALITY_W * z_quality + GROWTH_W * z_growth + TRUST_W * z_trust + RESONANCE_W * z_resonance + CONV_W * z_conv + IPO_W * z_ipo)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
