import numpy as np

CORE1_W = 0.28
CORE2_W = 0.24
GROW_QUAL_W = 0.16
GAP_W = 0.14
IPO_W = 0.10

CASH_S = 2.5
ROE_S = 2.0
CORE2_S = 1.8
GAP_S = 2.0
GROW_S = 1.5
QUAL_S = 2.0

__thesis__ = "质量驱动盈利动量与现金流确认的双核张力"


class Factor_20260531_005210_g32_1:
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
            cash_conf = panel['operating_cf_ps'] / np.maximum(np.abs(panel['eps']), 1e-8)
            ipo_premium = panel['open'] / np.maximum(panel['issue_price'], 1e-8)

        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1.0 / 3.0))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1.0 / 3.0))

        z_cash = _zscore(cash_conf)
        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)
        z_ey = _zscore(ey)
        z_cf = _zscore(cf_yield)
        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])

        cash_gate = 0.5 + 0.5 * np.tanh(z_cash * CASH_S)
        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_S)

        quality_factor = _zscore(z_roe + z_gm)
        quality_boost = 0.5 + 0.5 * np.tanh(quality_factor * QUAL_S)
        core1 = _zscore(z_ey * roe_gate * cash_gate * quality_boost)

        growth_raw = z_profit * z_rev
        growth_check = np.tanh(_zscore(growth_raw) * CORE2_S)
        core2 = _zscore(z_cf * growth_check * cash_gate)

        g_consistency = 1.0 - np.abs(np.tanh(z_profit) - np.tanh(z_rev))
        z_grow_qual = _zscore(g_consistency * cash_gate * (0.5 + 0.5 * np.tanh(z_profit * GROW_S)))

        z_gap_raw = _zscore(z_ey - z_cf)
        gap_signal = np.tanh(z_gap_raw * GAP_S)
        z_gap = _rank_norm(gap_signal * cash_gate)


        z_ipo = _rank_norm(-np.log(np.maximum(ipo_premium, 0.01)))

        score = (CORE1_W * core1 + CORE2_W * core2 + GROW_QUAL_W * z_grow_qual + GAP_W * z_gap + IPO_W * z_ipo)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
