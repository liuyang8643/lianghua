import numpy as np

CASH_ANCHOR_W = 0.24
VALUE_ANCHOR_W = 0.20
INTERACT_W = 0.16
Q_MOMENTUM_W = 0.14
GROWTH_COHERENCE_W = 0.12
MARGIN_QUALITY_W = 0.08

CASH_S = 2.5
ROE_S = 2.0
ACCRUAL_S = 2.0
LEVERAGE_S = 1.8
GROWTH_S = 1.5
CHAIN_S = 2.0
QM_S = 1.8
GM_S = 1.5

__thesis__ = "双锚验证：现金信任与盈利质量交互场合成因子"


class Factor_20260531_005210_g30_3:
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
            cash_coverage = panel['operating_cf_ps'] / np.maximum(np.abs(panel['eps']), 1e-8)
            accruals_raw = (panel['operating_cf_ps'] - panel['eps']) / panel['open']
            op_leverage = panel['gross_margin'] / np.maximum(np.abs(ey), 1e-6)

        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1.0 / 3.0))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1.0 / 3.0))

        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)
        z_ey = _zscore(ey)
        z_cf = _zscore(cf_yield)
        z_coverage = _zscore(cash_coverage)
        z_accruals = _zscore(accruals_raw)
        z_leverage = _zscore(op_leverage)
        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])


        # Anchor 1: Cash trust — accruals × leverage resonance × coverage certainty
        accrual_sig = np.tanh(z_accruals * ACCRUAL_S)
        leverage_sig = np.tanh(z_leverage * LEVERAGE_S)
        cash_anchor = _zscore(z_coverage * accrual_sig * leverage_sig)

        # Anchor 2: Value justified by profitability, gated by cash quality
        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_S)
        cash_gate = 0.5 + 0.5 * np.tanh(z_coverage * CASH_S)
        value_anchor = _zscore(z_ey * roe_gate * cash_gate)

        # Interaction field: where both anchors reinforce each other
        interact_field = _zscore(np.tanh(cash_anchor * value_anchor * CHAIN_S))

        # Quality momentum: ROE-driven earnings acceleration
        qm_raw = roe_s * panel['profit_yoy']
        qm_s = np.sign(qm_raw) * (np.abs(qm_raw) ** (1.0 / 3.0))
        z_qm = _zscore(qm_s)
        qm_sig = np.tanh(z_qm * QM_S)

        # Growth coherence: profit vs revenue directional agreement
        r_profit = _rank_norm(panel['profit_yoy'])
        r_rev = _rank_norm(panel['revenue_yoy'])
        growth_converge = np.tanh((1.0 - 0.5 * np.abs(r_profit - r_rev)) * GROWTH_S)
        z_growth = _zscore((z_profit + z_rev) * growth_converge)

        # Margin quality: GM scaled by cash trust
        margin_quality = _zscore(gm_s * cash_gate)
        margin_sig = np.tanh(margin_quality * GM_S)


        score = (CASH_ANCHOR_W * cash_anchor + VALUE_ANCHOR_W * value_anchor + INTERACT_W * interact_field + Q_MOMENTUM_W * qm_sig + GROWTH_COHERENCE_W * z_growth + MARGIN_QUALITY_W * margin_sig)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
