import numpy as np

PHASE_W = 0.20
COHERENCE_W = 0.18
CASH_REALITY_W = 0.16
VALUE_TRIPLE_W = 0.14
GROWTH_CASH_W = 0.12
EFFICIENCY_W = 0.10
LIFECYCLE_W = 0.10

PHASE_S = 2.0
CASH_S = 2.5
ROE_S = 2.0
GM_S = 2.0
GROWTH_S = 1.5
COHERENCE_S = 2.0
EFF_S = 2.0
CASH_RESIDUAL_ASYM = 1.5

__thesis__ = "现金流-增长相位套利与质量一致性锁定"


class Factor_20260531_005210_g39_1:
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
            ipo_premium = panel['open'] / panel['issue_price']

        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1/3))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1/3))

        z_cash = _zscore(cash_conf)
        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)
        z_ey = _zscore(ey)
        z_cf = _zscore(cf_yield)
        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])


        cash_gate = 0.5 + 0.5 * np.tanh(z_cash * CASH_S)
        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_S)
        gm_gate = 0.5 + 0.5 * np.tanh(z_gm * GM_S)
        triple_gate = cash_gate * roe_gate * gm_gate

        phase_angle = np.arctan2(z_profit, z_cash) / np.pi
        z_phase = _zscore(np.tanh(phase_angle * PHASE_S) * triple_gate)

        coherence_raw = 1.0 - np.abs(z_roe - z_gm) / (np.abs(z_roe) + np.abs(z_gm) + 1e-8)
        z_coherence = _zscore(np.tanh(coherence_raw * COHERENCE_S) * cash_gate)

        predicted = 0.5 * (z_roe + z_gm)
        cash_residual = z_cash - predicted
        asym = np.where(cash_residual < 0, cash_residual * CASH_RESIDUAL_ASYM, cash_residual * 0.5)
        z_cash_reality = _zscore(asym * cash_gate)

        z_value = _zscore(z_ey * triple_gate)

        z_growth = _zscore((z_profit + z_rev) * cash_gate * roe_gate)

        z_efficiency = _zscore(roe_s * gm_s * np.tanh(z_cf * EFF_S) * cash_gate)

        z_lifecycle = _rank_norm(np.where(np.isfinite(ipo_premium) & (ipo_premium > 0.01), -np.log(ipo_premium), np.nan) * (0.5 + 0.5 * np.tanh(z_roe * ROE_S)))

        score = (PHASE_W * z_phase + COHERENCE_W * z_coherence + CASH_REALITY_W * z_cash_reality + VALUE_TRIPLE_W * z_value + GROWTH_CASH_W * z_growth + EFFICIENCY_W * z_efficiency + LIFECYCLE_W * z_lifecycle)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
