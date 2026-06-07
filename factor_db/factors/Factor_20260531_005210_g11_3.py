import numpy as np

VALUE_W = 0.22
QUALITY_W = 0.16
RESONANCE_W = 0.14
GROWTH_W = 0.12
COHERENCE_W = 0.10
ACCRUAL_W = 0.10
IPO_W = 0.08

CASH_S = 2.5
COV_S = 2.0
MARGIN_S = 2.0
QUALITY_POS = 1.5
QUALITY_NEG = 3.0
GROWTH_S = 1.5
ACCRUAL_POS = 1.0
ACCRUAL_NEG = 3.0

__thesis__ = "三重信任乘性门控下的非对称质量价值共振"


class Factor_20260531_005210_g11_3:
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
            cash_cov = panel['operating_cf_ps'] / np.where(np.isfinite(panel['eps']) & (np.abs(panel['eps']) > 1e-8), np.abs(panel['eps']), np.nan)
            accruals = cf_yield - ey
            ipo_premium = panel['open'] / panel['issue_price']

        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1.0 / 3.0))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1.0 / 3.0))

        z_cf = _zscore(cf_yield)
        z_cov = _zscore(cash_cov)
        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)
        z_ey = _zscore(ey)
        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])

        # Triple trust: cash yield × coverage × margin — multiplicative credibility filter
        trust_cash = 0.5 + 0.5 * np.tanh(z_cf * CASH_S)
        trust_cov = 0.5 + 0.5 * np.tanh(z_cov * COV_S)
        trust_margin = 0.5 + 0.5 * np.tanh(z_gm * MARGIN_S)
        trust_triple = trust_cash * trust_cov * trust_margin

        # Quality: asymmetric tanh (mild reward, harsh penalty) gated by triple trust
        quality_raw = z_roe + z_gm
        z_quality_raw = _zscore(quality_raw)
        quality_asym = _asym_tanh(z_quality_raw, QUALITY_POS, QUALITY_NEG)
        z_quality = _zscore(quality_asym * trust_triple)

        # Value: earnings yield gated by triple trust
        z_value = _zscore(z_ey * trust_triple)

        # Resonance: nonlinear quality × value coupling amplified by cash trust
        resonance = np.tanh(z_quality * z_value)
        z_resonance = _zscore(resonance * trust_cash)

        # Growth: profit + revenue momentum confirmed by trust gate
        growth_raw = z_profit + z_rev
        growth_gate = 0.5 + 0.5 * np.tanh(growth_raw * GROWTH_S)
        z_growth = _zscore(growth_raw * growth_gate * trust_triple)

        # Cash-earnings coherence: alignment rewarded
        coherence = 1.0 - np.abs(z_cf - z_ey) * 0.5
        z_coherence = _zscore(coherence * trust_cash)

        # Accrual asymmetry: mild reward for cash>earnings, harsh penalty for earnings>cash
        accrual_score = _asym_tanh(accruals, ACCRUAL_POS, ACCRUAL_NEG)
        z_accrual = _zscore(accrual_score * trust_cov)

        # IPO discount: cheap relative to issue price, gated by credibility
        log_ipo = np.where(np.isfinite(ipo_premium) & (ipo_premium > 0.01), -np.log(ipo_premium), np.nan)
        z_ipo = _zscore(log_ipo * trust_triple)

        score = (VALUE_W * z_value + QUALITY_W * z_quality + RESONANCE_W * z_resonance + GROWTH_W * z_growth + COHERENCE_W * z_coherence + ACCRUAL_W * z_accrual + IPO_W * z_ipo)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
