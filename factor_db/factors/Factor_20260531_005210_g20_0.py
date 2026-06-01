import numpy as np

QUAL_RESONANCE_W = 0.22
TENSION_W = 0.20
GROWTH_CASH_W = 0.16
EFF_BRIDGE_W = 0.14
REV_CONF_W = 0.12
IPO_W = 0.08

RESONANCE_S = 2.5
TENSION_S = 2.0
GROWTH_S = 1.5
EFF_S = 2.0
REV_S = 1.5
IPO_S = 1.5

__thesis__ = "质量共振张力收敛与现金流成长双锚"


class Factor_20260531_005210_g20_0:
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
            cash_coverage = panel['operating_cf_ps'] / np.maximum(np.abs(panel['eps']), 1e-8)
            ipo_premium = panel['open'] / panel['issue_price']

        roe_cb = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1 / 3))
        gm_cb = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1 / 3))

        z_ey = _zscore(ey)
        z_cf = _zscore(cf_yield)
        z_cov = _zscore(cash_coverage)
        z_roe = _zscore(roe_cb)
        z_gm = _zscore(gm_cb)
        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])

        cash_gate = 0.5 + 0.5 * np.tanh(z_cov * TENSION_S)
        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * GROWTH_S)

        # Quality resonance: ROE × GM interaction with cash gate
        resonance_raw = np.tanh(z_roe * z_gm * RESONANCE_S)
        z_qual_resonance = _zscore(resonance_raw * cash_gate)

        # Value-cash tension: reward EY-CF alignment, penalize divergence
        tension_raw = z_ey * z_cf
        z_tension = _zscore(np.tanh(tension_raw * TENSION_S) * cash_gate)

        # Cash-backed growth: profit growth double-gated by cash and ROE
        growth_cash = z_profit * cash_gate * roe_gate
        z_growth_cash = _zscore(growth_cash)

        # Efficiency bridge: GM × CF interaction
        z_eff = _zscore(np.tanh(z_gm * z_cf * EFF_S))

        # Revenue confirmation: revenue growth needs GM and CF concurrence
        rev_conf_raw = z_rev * np.tanh(z_gm * REV_S) * np.tanh(z_cf * REV_S)
        z_rev_conf = _zscore(rev_conf_raw)

        # IPO premium with ROE quality gate
        ipo_raw = -np.log(np.maximum(ipo_premium, 0.01))
        z_ipo = _rank_norm(ipo_raw * (0.5 + 0.5 * np.tanh(z_roe * IPO_S)))

        score = (QUAL_RESONANCE_W * z_qual_resonance + TENSION_W * z_tension + GROWTH_CASH_W * z_growth_cash + EFF_BRIDGE_W * z_eff + REV_CONF_W * z_rev_conf + IPO_W * z_ipo)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
