import numpy as np

VALUE_W = 0.20
QUALITY_W = 0.18
GROWTH_W = 0.16
ACCRUAL_W = 0.12
HEALTH_W = 0.10
SPIRAL_W = 0.10
DIVERGENCE_W = 0.06
IPO_W = 0.04

CASH_GATE_S = 2.5
ROE_GATE_S = 2.0
ACCRUAL_S = 2.0
HEALTH_S = 2.0
SPIRAL_S = 1.5
GROWTH_EFF_S = 1.5
VALUE_BOOST_S = 2.0

__thesis__ = "应计质量双验证增长引擎与财务螺旋共振"


class Factor_20260531_005210_g8_3:
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
            cash_conf = panel['operating_cf_ps'] / np.where(np.isfinite(panel['eps']) & (np.abs(panel['eps']) > 1e-8), np.abs(panel['eps']), np.nan)
            accruals = cf_yield - ey
            growth_eff = panel['profit_yoy'] / np.where(np.isfinite(panel['revenue_yoy']) & (np.abs(panel['revenue_yoy']) > 1e-8), np.abs(panel['revenue_yoy']), np.nan)
            ipo_premium = panel['open'] / np.where(np.isfinite(panel['issue_price']) & (panel['issue_price'] > 1e-8), panel['issue_price'], np.nan)

        roe_cb = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1.0 / 3.0))
        gm_cb = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1.0 / 3.0))
        z_roe = _zscore(roe_cb)
        z_gm = _zscore(gm_cb)

        z_cash_conf = _zscore(cash_conf)
        cash_gate = 0.5 + 0.5 * np.tanh(z_cash_conf * CASH_GATE_S)
        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_GATE_S)

        # Quality: ROE + GM, cash-confirmed
        z_quality = _zscore((z_roe + z_gm) * cash_gate)

        # Value: EY with concavity boost + dual gate (ROE profitability + cash quality)
        z_ey = _zscore(ey)
        value_boost = 0.5 + 0.5 * np.tanh(np.abs(z_ey) * VALUE_BOOST_S)
        z_value = _zscore(z_ey * roe_gate * cash_gate * value_boost)

        # Growth: dual-engine organic (revenue × GM) + financial (profit × CF) + cross-product
        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])
        g_eff_sig = np.tanh(growth_eff * GROWTH_EFF_S)
        gm_confirm = 0.5 + 0.5 * np.tanh(z_gm * GROWTH_EFF_S)
        cf_confirm = 0.5 + 0.5 * np.tanh(z_cash_conf * GROWTH_EFF_S)
        organic_g = z_rev * gm_confirm
        financial_g = z_profit * cf_confirm
        cross_g = z_profit * z_rev * g_eff_sig
        z_growth = _zscore(organic_g + financial_g + cross_g)

        # Accrual quality: CF yield minus earnings yield signals earnings reliability
        z_accruals = _zscore(accruals)
        accruals_sig = np.tanh(z_accruals * ACCRUAL_S)

        # Financial health: cash coverage robustness
        health_sig = np.tanh(z_cash_conf * HEALTH_S)

        # 4-factor spiral: EY × CF yield × ROE × GM resonance
        z_cf = _zscore(cf_yield)
        spiral_raw = np.tanh(z_ey * z_cf * z_roe * z_gm * SPIRAL_S)
        z_spiral = _zscore(spiral_raw)

        # Asymmetric divergence: reward profit-led growth, discount revenue-only expansion
        divergence = z_profit - z_rev
        div_score = np.where(divergence > 0, divergence * 1.2, divergence * 0.6)
        z_divergence = _zscore(div_score)

        # IPO maturity: penalize recent IPOs near issue price
        z_ipo = _rank_norm(np.where(np.isfinite(ipo_premium) & (ipo_premium > 0.01), -np.log(ipo_premium), np.nan))

        score = (VALUE_W * z_value + QUALITY_W * z_quality + GROWTH_W * z_growth + ACCRUAL_W * accruals_sig + HEALTH_W * health_sig + SPIRAL_W * z_spiral + DIVERGENCE_W * z_divergence + IPO_W * z_ipo)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
