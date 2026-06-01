import numpy as np

VALUE_W = 0.22
TRUST_W = 0.18
GROWTH_DUAL_W = 0.16
ACCRUAL_SPIRAL_W = 0.14
EFF_DIV_W = 0.10
CASH_SIG_W = 0.08
IPO_W = 0.06

CASH_S = 2.0
TRUST_S = 2.0
ROE_GATE_S = 2.0
GROWTH_EFF_S = 1.5
ASYM_S = 0.5

__thesis__ = "应计信任螺旋与双引擎增长效率发散融合"


class Factor_20260531_005210_g8_1:
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
            cash_conf = panel['operating_cf_ps'] / np.maximum(np.abs(panel['eps']), 1e-8)
            accruals = cf_yield - ey
            growth_eff = panel['profit_yoy'] / np.maximum(np.abs(panel['revenue_yoy']), 1e-8)
            ipo_premium = panel['open'] / np.maximum(panel['issue_price'], 1e-8)

        # Cash quality: yield + coverage fused
        z_cf = _zscore(cf_yield)
        z_conf = _zscore(cash_conf)
        z_cash_quality = _zscore(z_cf + z_conf)
        cash_gate = 0.5 + 0.5 * np.tanh(z_cash_quality * CASH_S)

        # Earnings quality: cube-root ROE + GM
        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1/3))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1/3))
        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)
        z_earn_quality = _zscore(z_roe + z_gm)

        # Trust: quadratic cash-quality × earnings-quality interaction
        trust_raw = z_cash_quality * z_earn_quality
        z_trust = _zscore(np.tanh(trust_raw * TRUST_S))

        # Value: EY with triple anchor (ROE × cash × trust)
        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_GATE_S)
        trust_gate = 0.5 + 0.5 * np.tanh(z_trust)
        z_ey = _zscore(ey)
        z_value = _zscore(z_ey * roe_gate * cash_gate * trust_gate)

        # Dual growth engines: organic (rev × gm) + financial (profit × cash)
        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])
        gm_gate = 0.5 + 0.5 * np.tanh(z_gm)
        z_organic = _zscore(z_rev * gm_gate)
        z_financial = _zscore(z_profit * cash_gate)
        z_growth_dual = _zscore(z_organic + z_financial)

        # Accrual-trust spiral: accruals confirmed by trust
        z_accruals = _zscore(accruals)
        z_spiral = _zscore(z_accruals * trust_gate)

        # Growth efficiency divergence: asymmetric profit-rev spread
        g_eff_gate = 0.5 + 0.5 * np.tanh(growth_eff * GROWTH_EFF_S)
        divergence = z_profit - z_rev
        div_asym = np.where(divergence > 0, divergence, divergence * ASYM_S)
        z_div = _zscore(div_asym * g_eff_gate)

        # Direct cash quality signal
        cash_sig = np.tanh(z_cash_quality * CASH_S)

        # IPO maturity: favor stocks closer to issue price
        z_ipo = _rank_norm(-np.log(np.maximum(ipo_premium, 0.01)))

        score = (VALUE_W * z_value + TRUST_W * z_trust + GROWTH_DUAL_W * z_growth_dual + ACCRUAL_SPIRAL_W * z_spiral + EFF_DIV_W * z_div + CASH_SIG_W * cash_sig + IPO_W * z_ipo)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
