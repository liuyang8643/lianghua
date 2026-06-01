import numpy as np

CASCADE_W = 0.24
COHERENCE_W = 0.20
ACCRUAL_W = 0.16
EARN_QUAL_W = 0.14
LIFECYCLE_W = 0.12
DISPERSION_W = 0.06

CASH_S = 2.5
ROE_S = 2.0
GM_S = 1.8
CASCADE_S = 2.0
COH_S = 2.5
ACCRUAL_S = 2.0
EARN_S = 2.5
ASYM_P = 1.8
ASYM_N = 0.3
LIFE_S = 1.5

__thesis__ = "雁阵效应：三层级联×三信号相干×应计质量非对称验证"


class Factor_20260531_005210_g41_0:
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
            accrual_gap = (panel['operating_cf_ps'] - panel['eps']) / panel['open']
            ipo_ratio = panel['open'] / panel['issue_price']

        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1.0 / 3.0))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1.0 / 3.0))

        z_ey = _zscore(ey)
        z_cf = _zscore(cf_yield)
        z_cash = _zscore(cash_conf)
        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)
        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])
        z_accrual = _zscore(accrual_gap)

        cash_gate = 0.5 + 0.5 * np.tanh(z_cash * CASH_S)
        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_S)
        gm_gate = 0.5 + 0.5 * np.tanh(z_gm * GM_S)

        l1 = np.tanh(z_ey * z_cf * CASCADE_S) * cash_gate
        l2 = l1 * (0.5 + 0.5 * np.tanh((z_roe + z_gm) * CASCADE_S))
        l3 = l2 * (0.5 + 0.5 * np.tanh((z_profit + z_rev) * CASCADE_S))
        z_cascade = _zscore(l3)

        v_raw = z_ey * roe_gate * cash_gate
        q_raw = (z_roe + z_gm) * cash_gate * gm_gate
        g_raw = (z_profit + z_rev) * cash_gate
        z_v = _zscore(v_raw)
        z_q = _zscore(q_raw)
        z_g = _zscore(g_raw)

        coherence_raw = np.tanh(z_v * z_q * z_g * COH_S)
        z_coherence = _zscore(coherence_raw)

        acc_asym = np.where(z_accrual > 0, z_accrual * ASYM_P, z_accrual * ASYM_N)
        z_accrual_final = _zscore(np.tanh(acc_asym * ACCRUAL_S))

        earn_raw = np.tanh(z_cf * z_ey * EARN_S) * cash_gate
        z_earn_qual = _zscore(earn_raw)

        qual_level = np.tanh((z_roe + z_gm) * LIFE_S)
        life_raw = -np.log(np.maximum(ipo_ratio, 0.01))
        z_lifecycle = _rank_norm(life_raw * (0.7 + 0.3 * qual_level))

        r_v = _rank_norm(v_raw)
        r_q = _rank_norm(q_raw)
        r_g = _rank_norm(g_raw)
        mean_r = (r_v + r_q + r_g) / 3.0
        dispersion = np.sqrt(((r_v - mean_r)**2 + (r_q - mean_r)**2 + (r_g - mean_r)**2) / 3.0)
        z_dispersion = _zscore(-dispersion)


        score = (CASCADE_W * z_cascade + COHERENCE_W * z_coherence + ACCRUAL_W * z_accrual_final + EARN_QUAL_W * z_earn_qual + LIFECYCLE_W * z_lifecycle + DISPERSION_W * z_dispersion)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
