import numpy as np

VALUE_W = 0.26
QUALITY_W = 0.20
CASH_EFF_W = 0.16
GROWTH_W = 0.14
ACC_IPO_W = 0.10
TENSION_W = 0.06

QUALITY_ASYM_POS = 1.5
QUALITY_ASYM_NEG = 3.0
CASH_S = 2.5
GROWTH_S = 1.8
ACC_S = 2.0
TENSION_S = 1.5

__thesis__ = "非对称质量门控的乘性价值现金共振"


class Factor_20260531_005210_g1_0:
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

        def _asym_tanh(x, s_pos, s_neg):
            return np.where(x >= 0, np.tanh(x * s_pos), np.tanh(x * s_neg))

        with np.errstate(divide='ignore', invalid='ignore'):
            ey = panel['eps'] / panel['open']
            cf_yield = panel['operating_cf_ps'] / panel['open']
            ipo_premium = panel['open'] / panel['issue_price']

        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1.0 / 3.0))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1.0 / 3.0))
        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)
        quality_raw = z_roe + z_gm
        z_quality = _zscore(quality_raw)

        quality_asym = _asym_tanh(z_quality, QUALITY_ASYM_POS, QUALITY_ASYM_NEG)

        z_cf = _zscore(cf_yield)
        cf_quality_boost = 0.5 + 0.5 * np.tanh(z_quality * QUALITY_ASYM_POS)
        cf_efficient = z_cf * cf_quality_boost
        z_cf_efficient = _zscore(cf_efficient)

        z_ey = _zscore(ey)
        value_amplified = z_ey * (1.0 + 0.5 * quality_asym)
        z_value = _zscore(value_amplified)

        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])
        cf_gate = 0.5 + 0.5 * np.tanh(z_cf * CASH_S)
        margin_gate = 0.5 + 0.5 * np.tanh(z_gm * GROWTH_S)
        growth_confirmed = (z_profit + z_rev) * cf_gate * margin_gate
        z_growth = _zscore(growth_confirmed)

        accrual_raw = z_ey - z_cf
        z_accrual = _zscore(accrual_raw)
        accrual_score = -np.tanh(z_accrual * ACC_S)
        z_ipo = _rank_norm(-np.log(np.maximum(ipo_premium, 0.01)))
        acc_ipo = accrual_score + z_ipo
        z_acc_ipo = _zscore(acc_ipo)

        margin_tension = z_gm - z_cf
        z_tension = _zscore(margin_tension)
        tension_score = -np.tanh(z_tension * TENSION_S)


        score = (VALUE_W * z_value + QUALITY_W * quality_asym + CASH_EFF_W * z_cf_efficient + GROWTH_W * z_growth + ACC_IPO_W * z_acc_ipo + TENSION_W * tension_score)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
