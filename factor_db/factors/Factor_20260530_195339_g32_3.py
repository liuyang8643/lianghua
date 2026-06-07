import numpy as np

VALUE_W = 0.20
QUALITY_W = 0.18
GROWTH_QUAL_W = 0.16
CF_W = 0.14
CONVERGE_W = 0.12
STABILITY_W = 0.10
MOMENTUM_W = 0.06

CF_S = 2.0
ROE_S = 2.0
GROWTH_S = 1.5

__thesis__ = "质量动量与价值收敛的现金交织因子"


class Factor_20260530_195339_g32_3:
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
            coverage = panel['operating_cf_ps'] / np.where(np.isfinite(panel['eps']) & (np.abs(panel['eps']) > 1e-8), np.abs(panel['eps']), np.nan)

        # Cube-root transforms for outlier robustness
        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1/3))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1/3))

        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)

        # Quality core: ROE + gross_margin additive
        z_quality = _rank_norm(z_roe + z_gm)

        # Cash flow direct signal (tanh keeps continuous, no tie)
        z_cf_yield = _zscore(cf_yield)
        cf_sig = np.tanh(z_cf_yield * CF_S)

        # Value: earnings yield gated by ROE quality (single gate, not cascade)
        z_ey = _zscore(ey)
        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_S)
        z_value = _rank_norm(z_ey * roe_gate)

        # Quality-confirmed growth: profit+revenue momentum validated by ROE
        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])
        growth_gate = 0.5 + 0.5 * np.tanh(z_roe * GROWTH_S)
        z_growth_qual = _rank_norm((z_profit + z_rev) * growth_gate)

        # Value-CF convergence: quality value confirmed by cash
        z_converge = _rank_norm(z_value * cf_sig)

        # Cash stability: yield + coverage fused
        z_cov = _zscore(coverage)
        z_stability = _rank_norm(z_cf_yield + z_cov)

        # Revenue momentum
        z_momentum = _rank_norm(z_rev)

        score = (VALUE_W * z_value + QUALITY_W * z_quality + GROWTH_QUAL_W * z_growth_qual + CF_W * cf_sig + CONVERGE_W * z_converge + STABILITY_W * z_stability + MOMENTUM_W * z_momentum)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
