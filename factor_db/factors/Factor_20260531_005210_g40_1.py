import numpy as np

VALUE_W = 0.18
QUALITY_W = 0.18
GROWTH_W = 0.16
EFF_W = 0.14
AGREEMENT_W = 0.12
ACCRUAL_W = 0.10
LIFECYCLE_W = 0.07

CASH_S = 2.5
ROE_S = 2.0
EFF_S = 2.0
AGREEMENT_S = 1.5
ACCRUAL_S = 2.0
ASYM_POS = 1.6
ASYM_NEG = 0.4
ACC_ASYM_POS = 0.3
ACC_ASYM_NEG = 1.5

__thesis__ = "非对称共振网络：价值质量成长交叉验证"


class Factor_20260531_005210_g40_1:
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
            accrual_gap = (panel['operating_cf_ps'] - panel['eps']) / panel['open']

        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1.0 / 3.0))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1.0 / 3.0))

        z_cash = _zscore(cash_conf)
        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)
        z_ey = _zscore(ey)
        z_cf = _zscore(cf_yield)
        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])
        z_accrual = _zscore(accrual_gap)

        cash_gate = 0.5 + 0.5 * np.tanh(z_cash * CASH_S)
        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_S)

        z_value = _rank_norm(z_ey * roe_gate * cash_gate)

        q_raw = z_roe + z_gm
        q_asym = np.where(q_raw > 0, q_raw * ASYM_POS, q_raw * ASYM_NEG)
        z_quality = _rank_norm(q_asym * cash_gate)

        q_mom_raw = roe_s * panel['profit_yoy']
        q_mom_s = np.sign(q_mom_raw) * (np.abs(q_mom_raw) ** (1.0 / 3.0))
        z_q_mom = _zscore(q_mom_s)
        z_growth = _rank_norm((z_q_mom + z_rev) * cash_gate)

        op_eff = np.tanh(z_gm * z_cf * EFF_S)
        z_eff = _zscore(op_eff * cash_gate)

        agree_vq = np.tanh(z_value * z_quality * AGREEMENT_S)
        agree_vg = np.tanh(z_value * z_growth * AGREEMENT_S)
        agree_qg = np.tanh(z_quality * z_growth * AGREEMENT_S)
        z_agreement = _zscore((agree_vq + agree_vg + agree_qg) / 3.0)

        acc_asym = np.where(z_accrual > 0, z_accrual * ACC_ASYM_POS, z_accrual * ACC_ASYM_NEG)
        z_acc = _zscore(np.tanh(acc_asym * ACCRUAL_S))

        z_lifecycle = _rank_norm(np.where(np.isfinite(ipo_ratio) & (ipo_ratio > 0.01), -np.log(ipo_ratio), np.nan))


        score = (VALUE_W * z_value + QUALITY_W * z_quality + GROWTH_W * z_growth + EFF_W * z_eff + AGREEMENT_W * z_agreement + ACCRUAL_W * z_acc + LIFECYCLE_W * z_lifecycle)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
