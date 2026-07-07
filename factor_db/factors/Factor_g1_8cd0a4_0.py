import numpy as np

QUALITY_POWER = 0.5
VALUE_POWER = 1.0
GROWTH_BOOST = 0.5

class Factor_g1_8cd0a4_0:
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        base_valid = ~np.isnan(panel['close']) & (panel['close'] >= 2.0) & ~panel['st_mask']

        fin_ok = (np.isfinite(panel['roe']) & np.isfinite(panel['gross_margin']) &
                  np.isfinite(panel['profit_yoy']) & np.isfinite(panel['eps']))

        with np.errstate(divide='ignore', invalid='ignore'):
            ey = panel['eps'] / panel['close']

        valid_all = base_valid & fin_ok & np.isfinite(ey) & (panel['eps'] > 0)

        def _rank(x):
            nan = np.isnan(x)
            xx = np.where(nan, -np.inf, x.astype(np.float64))
            order = np.argsort(np.argsort(-xx, axis=1), axis=1).astype(np.float32)
            n = (~nan).sum(axis=1, keepdims=True).astype(np.float32)
            r = 1.0 - order / np.where(n > 0, n, 1.0)
            r[nan] = np.nan
            return r

        roe_c = np.where(valid_all, panel['roe'], np.nan)
        gm_c = np.where(valid_all, panel['gross_margin'], np.nan)
        pg_c = np.where(valid_all, panel['profit_yoy'], np.nan)
        ey_c = np.where(valid_all, ey, np.nan)

        r_roe = _rank(roe_c)
        r_gm = _rank(gm_c)
        r_pg = _rank(pg_c)
        r_ey = _rank(ey_c)

        quality = (r_roe ** QUALITY_POWER) * (r_gm ** QUALITY_POWER)
        value = r_ey ** VALUE_POWER
        growth = 1.0 + GROWTH_BOOST * r_pg

        score = quality * value * growth

        valid = base_valid & np.isfinite(score)
        return np.where(valid, score, np.nan)
