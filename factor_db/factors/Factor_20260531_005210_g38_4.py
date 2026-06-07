import numpy as np

VALUE_W = 0.22
QUALITY_W = 0.18
GROWTH_W = 0.15
EARN_QUAL_W = 0.14
CAP_EFF_W = 0.12
LIFECYCLE_W = 0.10

CASH_S = 2.5
ROE_S = 2.0
GROWTH_S = 1.5
EARN_QUAL_S = 2.0

__thesis__ = "盈利现金双确认叠加资本效率与生命周期锚定"


class Factor_20260531_005210_g38_4:
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
            ipo_ratio = panel['open'] / panel['issue_price']

        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1/3))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1/3))

        z_cash_conf = _zscore(cash_conf)
        cash_gate = 0.5 + 0.5 * np.tanh(z_cash_conf * CASH_S)

        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)
        z_quality = _zscore((z_roe + z_gm) * cash_gate)

        z_ey = _zscore(ey)
        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_S)
        z_value = _zscore(z_ey * roe_gate * cash_gate)

        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])
        g_gate = 0.5 + 0.5 * np.tanh((z_profit + z_rev) * GROWTH_S)
        z_growth = _zscore((z_profit + z_rev) * g_gate * cash_gate)

        z_cf = _zscore(cf_yield)
        earn_qual_raw = np.tanh(z_cf * z_ey * EARN_QUAL_S)
        z_earn_qual = _zscore(earn_qual_raw)

        cap_eff_raw = roe_s * gm_s
        cap_eff_s = np.sign(cap_eff_raw) * (np.abs(cap_eff_raw) ** (1/3))
        z_cap_eff = _zscore(cap_eff_s)

        z_lifecycle = _rank_norm(np.where(np.isfinite(ipo_ratio) & (ipo_ratio > 0.01), -np.log(ipo_ratio), np.nan))


        score = (VALUE_W * z_value + QUALITY_W * z_quality + GROWTH_W * z_growth + EARN_QUAL_W * z_earn_qual + CAP_EFF_W * z_cap_eff + LIFECYCLE_W * z_lifecycle)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
