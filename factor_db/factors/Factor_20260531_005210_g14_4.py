import numpy as np

VALUE_W = 0.24
QUALITY_W = 0.20
GROWTH_W = 0.18
CF_W = 0.16
IPO_STABILITY_W = 0.10
CONVERGE_W = 0.08

CASH_S = 2.5
ROE_S = 2.0
GM_S = 1.8
GROWTH_S = 1.5
TENSION_S = 1.5
IPO_S = 1.2

__thesis__ = "现金ROE梯度共识抑制质量张力的哑铃因子"


class Factor_20260531_005210_g14_4:
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
            ipo_prem = panel['open'] / panel['issue_price']

        z_cash = _zscore(cash_conf)
        cash_gate = 0.5 + 0.5 * np.tanh(z_cash * CASH_S)

        roe_s = np.sign(panel['roe']) * np.abs(panel['roe']) ** (1.0 / 3.0)
        z_roe = _zscore(roe_s)
        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_S)

        gm_s = np.sign(panel['gross_margin']) * np.abs(panel['gross_margin']) ** (1.0 / 3.0)
        z_gm = _zscore(gm_s)
        gm_gate = 0.5 + 0.5 * np.tanh(z_gm * GM_S)

        cash_roe_consensus = cash_gate * roe_gate
        z_consensus = _zscore(cash_roe_consensus)
        converge_sig = np.tanh(z_consensus * CASH_S)

        roe_gm_tension = z_roe - z_gm
        z_tension = _zscore(roe_gm_tension)
        tension_penalty = -np.tanh(z_tension * TENSION_S)

        z_ey = _zscore(ey)
        r_log_price = _rank_norm(np.log(np.maximum(panel['open'], 0.01)))

        z_value = _zscore(z_ey * cash_roe_consensus + 0.03 * r_log_price)

        quality_raw = (z_roe + z_gm) * cash_gate * gm_gate * (0.5 + 0.5 * tension_penalty)
        z_quality = _zscore(quality_raw + 0.03 * r_log_price)

        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])
        r_profit = _rank_norm(panel['profit_yoy'])
        r_rev = _rank_norm(panel['revenue_yoy'])
        growth_agree = 1.0 - 0.5 * np.abs(r_profit - r_rev)
        g_gate = 0.5 + 0.5 * np.tanh(growth_agree * GROWTH_S)
        growth_raw = (z_profit + z_rev) * g_gate * cash_gate
        z_growth = _zscore(growth_raw + 0.03 * r_log_price)

        z_cf = _zscore(cf_yield)
        cf_sig = np.tanh(z_cf * CASH_S)

        log_ipo = np.log(np.maximum(ipo_prem, 0.01))
        z_ipo = _zscore(log_ipo)
        ipo_bell = np.exp(-0.5 * (z_ipo * IPO_S) ** 2)
        r_share = _rank_norm(np.log(np.maximum(panel['total_share'], 1.0)))
        z_ipo_stable = _zscore(ipo_bell + 0.3 * r_share)


        score = (VALUE_W * z_value + QUALITY_W * z_quality + GROWTH_W * z_growth + CF_W * cf_sig + IPO_STABILITY_W * z_ipo_stable + CONVERGE_W * converge_sig)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
