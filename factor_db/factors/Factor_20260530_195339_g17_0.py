import numpy as np

VALUE_W = 0.22
TRUST_W = 0.20
ORG_GROWTH_W = 0.14
FIN_GROWTH_W = 0.10
MARGIN_MOM_W = 0.10
CF_EFF_W = 0.10
DIVERGENCE_W = 0.06

TRUST_S = 2.0
CF_S = 2.0
MOM_S = 1.5

__thesis__ = "现金质量互验与双引擎增长"


class Factor_20260530_195339_g17_0:
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        base_valid = ~np.isnan(panel['close']) & (panel['close'] >= 2.0) & ~panel['st_mask']

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
            ey = panel['eps'] / panel['close']
            cf_yield = panel['operating_cf_ps'] / panel['close']
            cash_coverage = panel['operating_cf_ps'] / np.where(np.isfinite(panel['eps']) & (np.abs(panel['eps']) > 1e-8), np.abs(panel['eps']), np.nan)

        z_cf_yield = _zscore(cf_yield)
        z_coverage = _zscore(cash_coverage)

        # Cash quality: yield + coverage fused
        cash_quality = z_cf_yield + z_coverage
        z_cash_quality = _zscore(cash_quality)
        cash_sig = np.tanh(z_cash_quality * CF_S)

        # Earnings quality: ROE + gross_margin fused
        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1/3))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1/3))
        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)
        earn_quality = z_roe + z_gm
        z_earn_quality = _zscore(earn_quality)

        # Cross-validation trust: cash quality x earnings quality (quadratic interaction)
        trust_raw = cash_quality * earn_quality
        z_trust = _zscore(trust_raw)

        # Organic growth: revenue confirmed by gross margin
        z_rev = _zscore(panel['revenue_yoy'])
        gm_confirm = 0.5 + 0.5 * np.tanh(z_gm * MOM_S)
        organic_growth = z_rev * gm_confirm
        z_organic = _zscore(organic_growth)

        # Financial growth: profit confirmed by cash coverage
        z_profit = _zscore(panel['profit_yoy'])
        cov_confirm = 0.5 + 0.5 * np.tanh(z_coverage * CF_S)
        financial_growth = z_profit * cov_confirm
        z_financial = _zscore(financial_growth)

        # Margin momentum: ROE/GM level x profit direction
        profit_dir = np.tanh(panel['profit_yoy'] * 0.5)
        mom_raw = (z_roe * profit_dir + z_gm * profit_dir) * 0.5
        z_mom = _zscore(mom_raw)

        # Cash-efficient value: EY confirmed by triple gate
        z_ey = _zscore(ey)
        triple_gate = 0.5 + 0.5 * np.tanh((z_earn_quality + z_cash_quality) * 0.5 * CF_S)
        z_value = _zscore(z_ey * triple_gate)

        # Asymmetric divergence: reward efficiency expansion, penalize margin dilution
        divergence = z_profit - z_rev
        div_score = np.where(divergence > 0, divergence * 1.0, divergence * 0.5)
        z_divergence = _zscore(div_score)


        score = (VALUE_W * z_value + TRUST_W * z_trust + ORG_GROWTH_W * z_organic + FIN_GROWTH_W * z_financial + MARGIN_MOM_W * z_mom + CF_EFF_W * cash_sig + DIVERGENCE_W * z_divergence)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
