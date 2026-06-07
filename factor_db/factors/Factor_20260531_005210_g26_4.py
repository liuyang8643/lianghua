import numpy as np

WEIGHT_RESONANCE = 0.22
WEIGHT_QUALITY = 0.20
WEIGHT_TRUST = 0.18
WEIGHT_GROWTH = 0.16
WEIGHT_RESIDUAL = 0.12
WEIGHT_IPO = 0.07

RESONANCE_S = 2.5
QUALITY_GATE_S = 2.0
CASH_GATE_S = 2.5
GROWTH_S = 1.8
RESIDUAL_ASYM = 0.5

__thesis__ = "现金盈利共振门控与成长分歧惩罚融合打分"


class Factor_20260531_005210_g26_4:
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
            cash_conf = panel['operating_cf_ps'] / np.where(np.isfinite(panel['eps']) & (np.abs(panel['eps']) > 1e-8), np.abs(panel['eps']), np.nan)
            ipo_premium = panel['open'] / np.where(np.isfinite(panel['issue_price']) & (panel['issue_price'] > 1e-8), panel['issue_price'], np.nan)

        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1/3))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1/3))

        z_ey = _zscore(ey)
        z_cf = _zscore(cf_yield)
        z_cash = _zscore(cash_conf)
        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)
        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])


        cash_gate = 0.5 + 0.5 * np.tanh(z_cash * CASH_GATE_S)
        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * QUALITY_GATE_S)

        resonance_raw = z_ey * z_cf
        resonance = np.tanh(resonance_raw * RESONANCE_S)
        z_resonance = _zscore(resonance * cash_gate * roe_gate)

        quality_product = z_roe * z_gm
        z_quality = _zscore(quality_product * cash_gate)

        earn_quality = z_roe + z_gm
        cash_quality = z_cf + z_cash
        trust_raw = earn_quality * cash_quality
        z_trust = _zscore(np.tanh(trust_raw * CASH_GATE_S) * cash_gate)

        gm_gate = 0.5 + 0.5 * np.tanh(z_gm * QUALITY_GATE_S)
        z_organic = _zscore(z_rev * gm_gate * cash_gate)
        z_financial = _zscore(z_profit * z_cash * roe_gate)
        div_penalty = 1.0 - 0.5 * np.abs(np.tanh(z_profit * GROWTH_S) - np.tanh(z_rev * GROWTH_S))
        z_growth = _zscore((z_organic + z_financial) * div_penalty)

        predicted_cash = _zscore(z_roe + z_gm)
        residual_cash = z_cash - predicted_cash
        asym_residual = np.where(residual_cash < 0, residual_cash * (1.0 + RESIDUAL_ASYM), residual_cash * 0.5)
        z_residual = _zscore(np.tanh(-asym_residual * CASH_GATE_S))

        z_ipo = _rank_norm(np.where(np.isfinite(ipo_premium) & (ipo_premium > 0.01), -np.log(ipo_premium), np.nan))

        score = (WEIGHT_RESONANCE * z_resonance + WEIGHT_QUALITY * z_quality + WEIGHT_TRUST * z_trust + WEIGHT_GROWTH * z_growth + WEIGHT_RESIDUAL * z_residual + WEIGHT_IPO * z_ipo)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
