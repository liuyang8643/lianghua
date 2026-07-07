import numpy as np

CF_W = 0.20
EY_CF_RESONANCE_W = 0.18
ROE_W = 0.15
GM_W = 0.12
GROWTH_W = 0.10
ACCRUAL_W = 0.08

CF_SIGMA = 1.8
ROE_SCALE = 1.5
GM_SCALE = 1.2
GR_SIGMA = 1.5
RES_SCALE = 1.5
ACC_SCALE = 1.0

__thesis__ = "现金流证实盈利的共振选股"


class Factor_20260530_195339_g3_2:
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        base_valid = ~np.isnan(panel['close']) & (panel['close'] >= 2.0) & ~panel['st_mask']

        def _zscore(x):
            x = x.astype(np.float64)
            mu = np.nanmean(x, axis=1, keepdims=True)
            sd = np.nanstd(x, axis=1, keepdims=True)
            return np.where(sd > 1e-12, (x - mu) / sd, np.nan).astype(np.float32)

        with np.errstate(divide='ignore', invalid='ignore'):
            cf_yield = panel['operating_cf_ps'] / panel['close']
            ey = panel['eps'] / panel['close']

        z_cf = _zscore(cf_yield)
        cash_score = np.tanh(z_cf * CF_SIGMA)

        z_ey = _zscore(ey)

        resonance_raw = z_ey * z_cf
        z_resonance = _zscore(resonance_raw)
        resonance_score = np.tanh(z_resonance * RES_SCALE)

        accrual_raw = z_ey - z_cf
        z_accrual = _zscore(accrual_raw)
        accrual_score = -np.tanh(z_accrual * ACC_SCALE)

        roe_signed = np.sign(panel['roe']) * np.sqrt(np.abs(panel['roe']))
        z_roe = _zscore(roe_signed)
        roe_score = np.tanh(z_roe * ROE_SCALE) * (0.5 + 0.5 * cash_score)

        z_gm = _zscore(panel['gross_margin'])
        gm_score = np.tanh(z_gm * GM_SCALE)

        z_profit = _zscore(panel['profit_yoy'])
        growth_score = np.tanh(z_profit * GR_SIGMA) * (0.5 + 0.5 * cash_score)



        score = (CF_W * cash_score + EY_CF_RESONANCE_W * resonance_score + ROE_W * roe_score + GM_W * gm_score + GROWTH_W * growth_score + ACCRUAL_W * accrual_score)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
