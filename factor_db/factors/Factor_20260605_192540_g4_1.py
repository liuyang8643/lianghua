import numpy as np

EY_W = 0.28
CF_W = 0.24
GM_W = 0.20
PROFIT_W = 0.14
REV_W = 0.10

__thesis__ = "盈利现金毛利三维锚定价值因子"


class Factor_20260605_192540_g4_1:
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

        ey_in = np.nan_to_num(ey, nan=0.0, posinf=0.0, neginf=0.0)
        cf_in = np.nan_to_num(cf_yield, nan=0.0, posinf=0.0, neginf=0.0)
        gm_in = np.nan_to_num(panel['gross_margin'], nan=0.0, posinf=0.0, neginf=0.0)
        profit_in = np.nan_to_num(panel['profit_yoy'], nan=0.0, posinf=0.0, neginf=0.0)
        rev_in = np.nan_to_num(panel['revenue_yoy'], nan=0.0, posinf=0.0, neginf=0.0)

        z_ey = _rank_norm(ey_in)
        z_cf = _rank_norm(cf_in)
        z_gm = _rank_norm(gm_in)
        z_profit = _rank_norm(profit_in)
        z_rev = _rank_norm(rev_in)

        cf_gate = 0.5 + 0.5 * np.tanh(z_cf)
        value_sig = z_ey * cf_gate

        eff_gate = 0.5 + 0.5 * np.tanh(z_cf + z_gm)
        eff_sig = z_gm * eff_gate

        growth_sig = np.tanh(z_profit * z_rev)

        score = EY_W * value_sig + CF_W * z_cf + GM_W * eff_sig + PROFIT_W * z_profit + REV_W * growth_sig

        return np.where(base_valid & np.isfinite(score), score, np.nan)
