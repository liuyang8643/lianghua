import numpy as np

QUALITY_W = 0.22
VALUE_W = 0.18
RESONANCE_W = 0.16
GROWTH_W = 0.12
EFFICIENCY_W = 0.10
FRAGILITY_W = 0.08
IPO_W = 0.08

QUALITY_POS_S = 1.8
QUALITY_NEG_S = 3.5
ROE_GATE_S = 2.0
CASH_GATE_S = 2.5
GROWTH_S = 1.5
RESONANCE_S = 1.5
FRAGILITY_S = 2.0

__thesis__ = "非对称质量门控的共振价值现金流融合因子"


class Factor_20260531_005210_g10_1:
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

        def _asym_tanh(x, s_pos, s_neg):
            return np.where(x >= 0, np.tanh(x * s_pos), np.tanh(x * s_neg))

        with np.errstate(divide='ignore', invalid='ignore'):
            ey = panel['eps'] / panel['open']
            cf_yield = panel['operating_cf_ps'] / panel['open']
            cash_conf = panel['operating_cf_ps'] / np.where(np.isfinite(panel['eps']) & (np.abs(panel['eps']) > 1e-8), np.abs(panel['eps']), np.nan)
            ipo_premium = panel['open'] / panel['issue_price']

        roe_t = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1.0 / 3.0))
        gm_t = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1.0 / 3.0))

        z_roe = _zscore(roe_t)
        z_gm = _zscore(gm_t)
        z_cash = _zscore(cash_conf)
        z_ey = _zscore(ey)
        z_cf = _zscore(cf_yield)
        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])

        cash_gate = 0.5 + 0.5 * np.tanh(z_cash * CASH_GATE_S)
        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_GATE_S)
        margin_gate = 0.5 + 0.5 * np.tanh(z_gm * GROWTH_S)

        # Quality: asymmetric tanh on ROE+GM additive, cash-gated
        quality_raw = z_roe + z_gm
        z_quality_raw = _zscore(quality_raw)
        z_quality = _zscore(_asym_tanh(z_quality_raw, QUALITY_POS_S, QUALITY_NEG_S) * cash_gate)

        # Value: EY gated by ROE profitability and cash confirmation
        z_value = _zscore(z_ey * roe_gate * cash_gate)

        # Resonance: earnings-cash flow alignment, quality-gated cross product
        resonance_raw = z_ey * z_cf
        z_resonance = _zscore(resonance_raw * (0.5 + 0.5 * np.tanh(z_quality_raw * RESONANCE_S)))

        # Growth: profit-revenue directional cross with dual gates
        z_growth = _zscore(z_profit * z_rev * cash_gate * margin_gate)

        # Efficiency: cash coverage standalone continuous signal
        efficiency_signal = np.tanh(z_cash * CASH_GATE_S)

        # Fragility: combined accrual divergence + margin-tension penalty
        accrual_raw = z_ey - z_cf
        tension_raw = z_gm - z_cf
        fragility_signal = -np.tanh(_zscore(accrual_raw + tension_raw) * FRAGILITY_S)

        # IPO: issue price premium rank
        ipo_signal = _rank_norm(np.where(np.isfinite(ipo_premium) & (ipo_premium > 0.01), -np.log(ipo_premium), np.nan))

        score = (QUALITY_W * z_quality + VALUE_W * z_value + RESONANCE_W * z_resonance + GROWTH_W * z_growth + EFFICIENCY_W * efficiency_signal + FRAGILITY_W * fragility_signal + IPO_W * ipo_signal)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
