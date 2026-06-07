import numpy as np

QUALITY_W = 0.35
VALUE_W = 0.30
GROWTH_W = 0.13
ACCRUAL_W = 0.12
PRICE_W = 0.10

__thesis__ = "质量价值乘积交互叠加增长应计价格锚"


class Factor_20260605_192540_g5_2:
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        base_valid = ~np.isnan(panel['open']) & (panel['open'] >= 2.0) & ~panel['st_mask']

        def _fillna_median(x):
            x = x.astype(np.float64)
            med = np.nanmedian(x, axis=1, keepdims=True)
            return np.where(np.isnan(x), med, x)

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
            accrual_gap = (panel['operating_cf_ps'] - panel['eps']) / panel['open']

        roe_f = _fillna_median(panel['roe'])
        gm_f = _fillna_median(panel['gross_margin'])
        ocf_f = _fillna_median(panel['operating_cf_ps'])
        profit_yoy_f = _fillna_median(panel['profit_yoy'])
        rev_yoy_f = _fillna_median(panel['revenue_yoy'])
        ey_f = _fillna_median(ey)
        cfy_f = _fillna_median(cf_yield)
        ag_f = _fillna_median(accrual_gap)

        roe_s = np.sign(roe_f) * np.abs(roe_f) ** (1.0 / 3.0)
        gm_s = np.sign(gm_f) * np.abs(gm_f) ** (1.0 / 3.0)

        r_roe = _rank_norm(roe_s)
        r_gm = _rank_norm(gm_s)
        r_ocf = _rank_norm(ocf_f)
        quality = (r_roe + r_gm + r_ocf) / 3.0

        r_ey = _rank_norm(ey_f)
        r_cfy = _rank_norm(cfy_f)
        value = (r_ey + r_cfy) / 2.0

        r_profit = _rank_norm(profit_yoy_f)
        r_rev = _rank_norm(rev_yoy_f)
        growth = r_profit * r_rev
        growth = np.sign(growth) * np.sqrt(np.abs(growth))

        r_accrual = _rank_norm(ag_f)

        r_price_inv = _rank_norm(-panel['open'])

        qv = quality * value
        qv_s = np.sign(qv) * np.sqrt(np.abs(qv))

        score = (QUALITY_W * qv_s + VALUE_W * value + GROWTH_W * growth
                 + ACCRUAL_W * r_accrual + PRICE_W * r_price_inv)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
