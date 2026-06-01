import numpy as np

VALUE_W = 0.35
QUALITY_W = 0.25
GROWTH_W = 0.18
CF_W = 0.12
IPO_W = 0.03

QUALITY_S = 1.5

__thesis__ = "综合质量校准的估值成长双驱因子"


class Factor_20260530_195339_g8_4:
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        base_valid = ~np.isnan(panel['open']) & (panel['open'] >= 2.0) & ~panel['st_mask']

        def _zscore(x):
            x = x.astype(np.float64)
            mu = np.nanmean(x, axis=1, keepdims=True)
            sd = np.nanstd(x, axis=1, keepdims=True)
            return np.where(sd > 1e-12, (x - mu) / sd, np.nan).astype(np.float32)

        with np.errstate(divide='ignore', invalid='ignore'):
            ey = panel['eps'] / panel['open']
            cf_yield = panel['operating_cf_ps'] / panel['open']
            ipo_premium = panel['open'] / panel['issue_price']

        # Quality composite: profitability (ROE) + pricing power (GM) + cash realization (CF yield)
        roe_s = np.sign(panel['roe']) * np.sqrt(np.abs(panel['roe']))
        z_quality = _zscore(_zscore(roe_s) + _zscore(panel['gross_margin']) + _zscore(cf_yield))

        # Quality-calibrated value: EY weighted by quality confidence
        quality_conf = 0.5 + 0.5 * np.tanh(z_quality * QUALITY_S)
        z_value = _zscore(ey * quality_conf)

        # Quality-confirmed growth
        z_growth = _zscore(_zscore(panel['profit_yoy']) + _zscore(panel['revenue_yoy']))
        z_growth_confirmed = _zscore(z_growth * quality_conf)

        # Pure cash yield signal
        z_cf = _zscore(cf_yield)
        cf_sig = np.tanh(z_cf * 2.0)

        z_ipo = _zscore(-np.log(np.maximum(ipo_premium, 0.01)))

        score = (VALUE_W * z_value + QUALITY_W * z_quality + GROWTH_W * z_growth_confirmed + CF_W * cf_sig + IPO_W * z_ipo)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
