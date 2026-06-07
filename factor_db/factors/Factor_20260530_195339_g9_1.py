import numpy as np

VALUE_W = 0.30
QUALITY_W = 0.22
GROWTH_W = 0.18
CF_W = 0.14
IPO_W = 0.06

Q_AMP = 1.5

__thesis__ = "质量放大估值的秩标准化多因子融合"


class Factor_20260530_195339_g9_1:
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

        with np.errstate(divide='ignore', invalid='ignore'):
            ey = panel['eps'] / panel['open']
            cf_yield = panel['operating_cf_ps'] / panel['open']
            ipo_premium = panel['open'] / panel['issue_price']

        roe_s = np.sign(panel['roe']) * np.sqrt(np.abs(panel['roe']))
        r_roe = _rank_norm(roe_s)
        r_gm = _rank_norm(panel['gross_margin'])
        r_quality = _rank_norm(r_roe + r_gm)

        q_mult = 1.0 + Q_AMP * np.maximum(r_quality, 0.0)
        r_value = _rank_norm(ey * q_mult)

        r_cf = _rank_norm(cf_yield)

        r_growth = _rank_norm(
            _rank_norm(panel['revenue_yoy']) + _rank_norm(panel['profit_yoy'])
        )

        r_ipo = _rank_norm(np.where(np.isfinite(ipo_premium) & (ipo_premium > 0.01), -np.log(ipo_premium), np.nan))

        score = (VALUE_W * r_value + QUALITY_W * r_quality + GROWTH_W * r_growth + CF_W * r_cf + IPO_W * r_ipo)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
