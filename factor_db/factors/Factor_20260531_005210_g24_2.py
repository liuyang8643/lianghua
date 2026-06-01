import numpy as np

TRUST_W = 0.20
CONSENSUS_W = 0.18
CONVEXITY_W = 0.16
GROWTH_W = 0.14
FRAGILITY_W = 0.12
IPO_W = 0.10

TRUST_S = 2.0
CASH_GATE_S = 2.5
ROE_GATE_S = 2.0
CONSENSUS_S = 1.5
CONVEXITY_S = 1.5
GROWTH_S = 1.8
FRAGILITY_S = 2.5
ASYM_S = 0.6

__thesis__ = "信任共识凸性价值与双引擎增长脆弱性融合"


class Factor_20260531_005210_g24_2:
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
            ipo_premium = panel['open'] / np.maximum(panel['issue_price'], 1e-8)

        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1/3))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1/3))

        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)
        z_ey = _zscore(ey)
        z_cf = _zscore(cf_yield)
        z_cash_conf = _zscore(cash_conf)
        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])

        # Gates
        cash_gate = 0.5 + 0.5 * np.tanh(z_cash_conf * CASH_GATE_S)
        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_GATE_S)

        # Trust: quadratic cash-quality × earnings-quality bridge, gated
        earn_quality = z_roe + z_gm
        cash_quality = z_cf + z_cash_conf
        trust_raw = cash_quality * earn_quality
        trust_gate = 0.5 + 0.5 * np.tanh(trust_raw * TRUST_S / 2.0)
        z_trust = _zscore(np.tanh(trust_raw * TRUST_S) * cash_gate * roe_gate)

        # Consensus: pairwise agreement among three quality signals as diversity bonus
        t_roe = np.tanh(z_roe)
        t_gm = np.tanh(z_gm)
        t_cash = np.tanh(z_cash_conf)
        consensus_pairwise = (t_roe * t_gm + t_roe * t_cash + t_gm * t_cash) / 3.0
        z_consensus = _zscore(consensus_pairwise * trust_gate)

        # Convexity: EY×CF joint magnitude amplifies their sum, convex payoff
        joint_mag = np.sqrt(np.maximum(z_ey ** 2 + z_cf ** 2, 1e-12))
        convexity_raw = np.tanh((z_ey + z_cf) * joint_mag * CONVEXITY_S)
        z_convexity = _zscore(convexity_raw * cash_gate * roe_gate)

        # Dual growth: organic engine (revenue×quality) + financial engine (profit×cash confidence)
        gm_gate = 0.5 + 0.5 * np.tanh(z_gm * GROWTH_S / 1.5)
        z_organic = _zscore(z_rev * gm_gate * cash_gate)
        z_financial = _zscore(z_profit * z_cash_conf * roe_gate)
        z_growth = _zscore((z_organic + z_financial) * trust_gate)

        # Fragility: asymmetric penalty when EY exceeds CF (accrual risk)
        accrual_signal = z_ey - z_cf
        asym_fragile = np.where(accrual_signal < 0, accrual_signal * (1.0 + ASYM_S), accrual_signal * 0.5)
        z_fragility = _zscore(np.tanh(-asym_fragile * FRAGILITY_S))

        # IPO fade: favor stocks nearer to issue price
        z_ipo = _rank_norm(-np.log(np.maximum(ipo_premium, 0.01)))

        score = (TRUST_W * z_trust + CONSENSUS_W * z_consensus + CONVEXITY_W * z_convexity + GROWTH_W * z_growth + FRAGILITY_W * z_fragility + IPO_W * z_ipo)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
