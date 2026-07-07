import numpy as np

VALUE_W = 0.26
QUALITY_W = 0.22
GROWTH_W = 0.18
CF_W = 0.14
IPO_W = 0.06

Q_AMP_S = 1.2
CF_GATE_S = 1.5

__thesis__ = "质量置信对称放大与现金流修正的多因子融合"


class Factor_20260530_195339_g10_0:
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
            ipo_premium = panel['close'] / panel['issue_price']

        roe_s = np.sign(panel['roe']) * np.sqrt(np.abs(panel['roe']))
        r_roe = _rank_norm(roe_s)
        r_gm = _rank_norm(panel['gross_margin'])
        r_quality = _rank_norm(r_roe + r_gm)

        r_cf = _rank_norm(cf_yield)
        cf_confidence = 0.5 + 0.5 * np.tanh(r_cf * CF_GATE_S)
        r_quality_adj = _rank_norm(r_quality * cf_confidence)

        quality_amp = 1.0 + Q_AMP_S * np.tanh(r_quality_adj * 0.8)
        z_ey = _zscore(ey)
        r_value = _rank_norm(z_ey * quality_amp)

        r_rev = _rank_norm(panel['revenue_yoy'])
        r_profit = _rank_norm(panel['profit_yoy'])
        quality_conf = 0.5 + 0.5 * np.tanh(r_quality * 2.0)
        r_growth = _rank_norm(r_rev * (1.0 + 0.3 * quality_conf) + r_profit)

        cf_sig = _rank_norm(np.tanh(r_cf * 2.0))

        r_ipo = _rank_norm(np.where(np.isfinite(ipo_premium) & (ipo_premium > 0.01), -np.log(ipo_premium), np.nan))

        score = (VALUE_W * r_value + QUALITY_W * r_quality_adj + GROWTH_W * r_growth + CF_W * cf_sig + IPO_W * r_ipo)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
