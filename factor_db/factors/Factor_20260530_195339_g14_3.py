import numpy as np

VALUE_W = 0.27
QUALITY_W = 0.23
GROWTH_W = 0.17
CF_W = 0.14
MARGIN_W = 0.08
INTERACT_W = 0.06
IPO_W = 0.04

ROE_GATE_S = 2.0
CF_GATE_S = 1.8

__thesis__ = "立根稳健化与门控价值质量融合"


class Factor_20260530_195339_g14_3:
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
            ipo_premium = panel['open'] / panel['issue_price']

        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1/3))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1/3))
        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)
        z_quality = _zscore(z_roe + z_gm)

        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_GATE_S)
        z_ey = _zscore(ey)
        z_value = _zscore(z_ey * roe_gate)

        r_g = _rank_norm(panel['profit_yoy']) + _rank_norm(panel['revenue_yoy'])
        z_growth = _zscore(r_g)

        margin_exp = _zscore(panel['profit_yoy'] - panel['revenue_yoy'])

        r_cf = _rank_norm(cf_yield)
        cf_sig = np.tanh(r_cf * CF_GATE_S)

        z_interact = _zscore(z_quality * _rank_norm(ey))

        z_ipo = _rank_norm(-np.log(np.maximum(ipo_premium, 0.01)))

        score = (VALUE_W * z_value + QUALITY_W * z_quality + GROWTH_W * z_growth + CF_W * cf_sig + MARGIN_W * margin_exp + INTERACT_W * z_interact + IPO_W * z_ipo)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
