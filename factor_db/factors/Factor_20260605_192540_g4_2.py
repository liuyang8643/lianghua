import numpy as np

QUALITY_SPREAD_S = 2.5
VALUE_MISPRICE_S = 2.0
GROWTH_S = 1.8
SIZE_ANCHOR = 0.015

__thesis__ = "质优价廉截面错位——买财务强但市场定价低的股票"


class Factor_20260605_192540_g4_2:
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        base_valid = ~np.isnan(panel['open']) & (panel['open'] >= 2.0) & ~panel['st_mask']

        def _zscore(x):
            x = x.astype(np.float64)
            mu = np.nanmean(x, axis=1, keepdims=True)
            sd = np.nanstd(x, axis=1, keepdims=True)
            return np.where(sd > 1e-12, (x - mu) / sd, 0.0).astype(np.float32)

        def _na0(x):
            return np.where(np.isfinite(x), x, 0.0)

        with np.errstate(divide='ignore', invalid='ignore'):
            ey = panel['eps'] / panel['open']
            cf_yield = panel['operating_cf_ps'] / panel['open']
            accrual_gap = (panel['operating_cf_ps'] - panel['eps']) / panel['open']
            ipo_premium = panel['open'] / np.where(np.isfinite(panel['issue_price']) & (panel['issue_price'] > 1e-8), panel['issue_price'], np.nan)

        roe_s = np.sign(panel['roe']) * np.abs(panel['roe']) ** (1.0 / 3.0)
        gm_s = np.sign(panel['gross_margin']) * np.abs(panel['gross_margin']) ** (1.0 / 3.0)

        size_log = np.log(panel['total_share'] * panel['open'])

        z_roe = _na0(_zscore(roe_s))
        z_gm = _na0(_zscore(gm_s))
        z_cf = _na0(_zscore(cf_yield))
        z_ey = _na0(_zscore(ey))
        z_accrual = _na0(_zscore(accrual_gap))
        z_profit = _na0(_zscore(panel['profit_yoy']))
        z_rev = _na0(_zscore(panel['revenue_yoy']))
        z_lifecycle = _na0(_zscore(np.where(np.isfinite(ipo_premium) & (ipo_premium > 0.01), -np.log(ipo_premium), np.nan)))
        z_size = _na0(_zscore(size_log))

        quality_raw = z_roe + z_gm + z_cf + z_accrual
        quality_gate = 0.5 + 0.5 * np.tanh((z_roe + z_gm) * QUALITY_SPREAD_S)
        z_quality = _zscore(quality_raw * quality_gate + SIZE_ANCHOR * z_size)

        value_raw = z_ey + z_cf + z_lifecycle
        accrual_penalty = np.tanh(z_accrual * VALUE_MISPRICE_S)
        z_value = _zscore(value_raw * accrual_penalty + SIZE_ANCHOR * z_size)

        growth_raw = z_profit * z_rev
        growth_gate = 0.5 + 0.5 * np.tanh((z_profit + z_rev) * GROWTH_S)
        z_growth = _zscore(growth_raw * growth_gate + SIZE_ANCHOR * z_size)

        quality_sig = np.tanh(z_quality * QUALITY_SPREAD_S)
        value_sig = np.tanh(z_value * VALUE_MISPRICE_S)
        growth_sig = np.tanh(z_growth * GROWTH_S)

        interaction = quality_sig * value_sig * (0.5 + 0.5 * growth_sig)

        score = _zscore(interaction + SIZE_ANCHOR * z_size)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
