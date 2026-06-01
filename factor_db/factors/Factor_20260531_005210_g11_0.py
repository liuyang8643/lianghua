import numpy as np

VALUE_W = 0.20
QUALITY_W = 0.18
GROWTH_W = 0.14
COHERENCE_W = 0.12
CASH_W = 0.12
IPO_W = 0.08
DIVERGE_W = 0.08

COH_S = 2.5
QUAL_S_POS = 1.5
QUAL_S_NEG = 3.0
CASH_S = 2.0
GROWTH_S = 1.8
DIVERGE_S = 2.0
IPO_S = 1.5
COH_GATE_S = 2.0

__thesis__ = "非对称相干三角门控的价值质量现金共振"


class Factor_20260531_005210_g11_0:
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
            ipo_premium = panel['open'] / panel['issue_price']
            growth_eff = panel['profit_yoy'] / np.maximum(np.abs(panel['revenue_yoy']), 1e-8)

        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1.0 / 3.0))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1.0 / 3.0))

        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)
        z_ey = _zscore(ey)
        z_cf = _zscore(cf_yield)
        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])
        z_ge = _zscore(growth_eff)

        quality_raw = z_roe + z_gm
        z_quality = _zscore(quality_raw)
        quality_asym = _asym_tanh(z_quality, QUAL_S_POS, QUAL_S_NEG)

        coherence_raw = 1.0 - 0.5 * np.abs(z_ey - z_cf)
        z_coherence = _zscore(coherence_raw)
        coherence_gate = 0.5 + 0.5 * _asym_tanh(z_coherence, COH_GATE_S, COH_GATE_S * 1.5)

        z_value = _zscore(z_ey * (0.5 + 0.5 * quality_asym) * coherence_gate)

        cash_gate = 0.5 + 0.5 * np.tanh(z_cf * CASH_S)
        z_quality_gated = _zscore((z_roe + z_gm) * cash_gate * coherence_gate)

        growth_qual_gate = 0.5 + 0.5 * np.tanh(z_quality * GROWTH_S)
        growth_eff_gate = 0.5 + 0.5 * np.tanh(z_ge * GROWTH_S)
        growth_raw = z_profit + z_rev
        z_growth = _zscore(growth_raw * coherence_gate * growth_qual_gate * growth_eff_gate)

        cash_quality = _zscore(z_cf * (1.0 + 0.5 * quality_asym))

        log_ipo = -np.log(np.maximum(ipo_premium, 0.01))
        z_ipo = _zscore(log_ipo * cash_gate)

        margin_cash_gap = z_gm - z_cf
        z_gap = _zscore(margin_cash_gap)
        diverge_score = -_asym_tanh(z_gap, DIVERGE_S * 0.5, DIVERGE_S)


        score = (VALUE_W * z_value + QUALITY_W * z_quality_gated + GROWTH_W * z_growth + COHERENCE_W * z_coherence + CASH_W * cash_quality + IPO_W * z_ipo + DIVERGE_W * diverge_score)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
