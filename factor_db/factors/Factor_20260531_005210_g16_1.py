import numpy as np

CRED_QUALITY_W = 0.26
VG_CONSENSUS_W = 0.20
ACCRUAL_W = 0.12
GM_CF_W = 0.12
MOM_W = 0.10
IPO_W = 0.06
INTERACT_W = 0.08

ROE_S = 2.0
CASH_S = 2.5
ACC_S = 1.5
GM_S = 2.0
MOM_S = 1.8
CRED_S = 2.0

__thesis__ = "可信度缩放价值增长双锚乘法共识"


class Factor_20260531_005210_g16_1:
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        base_valid = ~np.isnan(panel['close']) & (panel['close'] >= 2.0) & ~panel['st_mask']

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
            ey = panel['eps'] / panel['close']
            cf_yield = panel['operating_cf_ps'] / panel['close']
            cash_conf = panel['operating_cf_ps'] / np.where(np.isfinite(panel['eps']) & (np.abs(panel['eps']) > 1e-8), np.abs(panel['eps']), np.nan)
            ipo_premium = panel['close'] / panel['issue_price']

        z_ey = _zscore(ey)
        z_cf = _zscore(cf_yield)
        z_cash_conf = _zscore(cash_conf)
        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])

        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1/3))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1/3))
        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)

        cred_raw = z_cash_conf * z_roe * z_gm
        z_cred = _rank_norm(cred_raw)

        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_S)
        cash_gate = 0.5 + 0.5 * np.tanh(z_cash_conf * CASH_S)
        eff_gate = 0.5 + 0.5 * np.tanh((z_profit - z_rev) * ACC_S)

        value_anchor = _zscore(z_ey * roe_gate * cash_gate)
        growth_anchor = _zscore((z_profit + z_rev) * cash_gate * eff_gate)

        vg_consensus = _rank_norm(value_anchor * growth_anchor)

        cred_quality = _zscore(z_cred * vg_consensus * np.tanh(z_roe * CRED_S))

        accrual_raw = z_ey - z_cf
        accrual = -np.tanh(accrual_raw * ACC_S)

        gm_cf = np.tanh(z_gm * GM_S) * np.tanh(z_cf * CASH_S)

        rev_mom = _rank_norm(z_rev)

        z_ipo = _rank_norm(np.where(np.isfinite(ipo_premium) & (ipo_premium > 0.01), -np.log(ipo_premium), np.nan))


        z_interact = _zscore(value_anchor * z_gm * np.tanh(z_cf * CASH_S))

        score = (CRED_QUALITY_W * cred_quality + VG_CONSENSUS_W * vg_consensus + ACCRUAL_W * accrual + GM_CF_W * gm_cf + MOM_W * rev_mom + IPO_W * z_ipo + INTERACT_W * z_interact)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
