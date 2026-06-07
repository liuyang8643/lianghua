import numpy as np

VALUE_W = 0.24
QUALITY_W = 0.22
GROWTH_W = 0.18
CF_W = 0.13
ACCRUAL_W = 0.10
INTERACT_W = 0.07

CASH_GATE_S = 2.5
ROE_GATE_S = 2.0
ACCRUAL_S = 2.0

__thesis__ = "现金锚定质价增长秩归一化交叉融合"


class Factor_20260531_005210_g23_3:
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
            accruals = cf_yield - ey

        z_cash = _zscore(cash_conf)
        cash_gate = 0.5 + 0.5 * np.tanh(z_cash * CASH_GATE_S)

        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1/3))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1/3))
        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)
        quality_raw = (z_roe + z_gm) * cash_gate
        z_quality = _rank_norm(quality_raw)

        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_GATE_S)
        z_ey = _zscore(ey)
        value_raw = z_ey * roe_gate * cash_gate
        z_value = _zscore(value_raw)

        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])
        growth_raw = (z_profit + z_rev) * cash_gate
        z_growth = _rank_norm(growth_raw)

        z_cf = _zscore(cf_yield)
        cf_sig = np.tanh(z_cf * CASH_GATE_S)

        z_acc = _zscore(accruals)
        accrual_sig = np.tanh(z_acc * ACCRUAL_S)

        z_interact = _zscore(z_quality * z_growth)


        score = (VALUE_W * z_value + QUALITY_W * z_quality + GROWTH_W * z_growth + CF_W * cf_sig + ACCRUAL_W * accrual_sig + INTERACT_W * z_interact)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
