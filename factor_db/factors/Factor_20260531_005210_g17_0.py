import numpy as np

VALUE_W = 0.26
GROWTH_W = 0.20
OP_EFF_W = 0.16
MARGIN_W = 0.14
ACCRUAL_W = 0.12

ROE_S = 2.0
CF_S = 2.5
EFF_S = 2.0
GROWTH_S = 1.5
MARGIN_S = 2.0
ACCRUAL_S = 2.0

__thesis__ = "经营效率桥接的质量价值双螺旋因子"


class Factor_20260531_005210_g17_0:
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

        roe_s = np.sign(panel['roe']) * np.abs(panel['roe']) ** (1.0 / 3.0)
        gm_s = np.sign(panel['gross_margin']) * np.abs(panel['gross_margin']) ** (1.0 / 3.0)

        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)
        z_ey = _zscore(ey)
        z_cf = _zscore(cf_yield)
        z_cash = _zscore(cash_conf)
        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])

        op_eff = np.tanh(z_gm * z_cf * EFF_S)

        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_S)
        cash_gate = 0.5 + 0.5 * np.tanh(z_cash * CF_S)
        z_value = _rank_norm(z_ey * roe_gate * cash_gate * op_eff)

        growth_gate = 0.5 + 0.5 * np.tanh(z_profit * GROWTH_S)
        z_growth = _rank_norm((z_profit + z_rev) * op_eff * growth_gate)

        z_op_eff = _zscore(op_eff * z_gm)

        margin_sig = np.tanh(z_gm * cash_gate * MARGIN_S)

        accrual_gap = cf_yield - ey
        z_accrual = _zscore(accrual_gap)
        accrual_sig = np.tanh(z_accrual * ACCRUAL_S)


        score = (VALUE_W * z_value + GROWTH_W * z_growth + OP_EFF_W * z_op_eff + MARGIN_W * margin_sig + ACCRUAL_W * accrual_sig)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
