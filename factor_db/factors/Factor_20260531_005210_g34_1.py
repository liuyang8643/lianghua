import numpy as np

VALUE_W = 0.24
QUALITY_W = 0.20
GROWTH_W = 0.18
CF_W = 0.14
ALIGN_W = 0.10
IPO_W = 0.08

CASH_GATE_S = 2.5
ROE_GATE_S = 2.0
GROWTH_S = 1.8
CF_S = 2.0
ALIGN_S = 1.5

__thesis__ = "现金验证盈利质量增长对齐的价值发现"


class Factor_20260531_005210_g34_1:
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
            ipo_premium = panel['close'] / np.where(np.isfinite(panel['issue_price']) & (panel['issue_price'] > 1e-8), panel['issue_price'], np.nan)

        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1/3))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1/3))

        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)
        z_ey = _zscore(ey)
        z_cf = _zscore(cf_yield)
        z_cash_conf = _zscore(cash_conf)
        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])

        cash_gate = 0.5 + 0.5 * np.tanh(z_cash_conf * CASH_GATE_S)
        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_GATE_S)

        # Value: EY triple-gated by cash, ROE, and gross margin quality
        gm_gate = 0.5 + 0.5 * np.tanh(z_gm * ROE_GATE_S / 2.0)
        z_value = _zscore(z_ey * cash_gate * roe_gate * gm_gate)

        # Quality: ROE + GM composite, cash-and-profitability verified
        quality_raw = z_roe + z_gm
        z_quality = _zscore(quality_raw * cash_gate * roe_gate)

        # Growth: signed-sqrt profit×revenue cross term, cash-gated
        growth_raw = np.sign(z_profit * z_rev) * np.sqrt(np.abs(z_profit * z_rev))
        growth_gate = 0.5 + 0.5 * np.tanh(z_rev * GROWTH_S / 2.0)
        z_growth = _zscore(growth_raw * cash_gate * growth_gate)

        # CF Premium: non-linear operating cash flow yield
        cf_tanh = np.tanh(z_cf * CF_S)
        z_cf_premium = _zscore(cf_tanh * cash_gate)

        # Growth-Quality Alignment: profit and revenue tell consistent story
        align_raw = 1.0 - np.abs(np.tanh((z_profit - z_rev) * ALIGN_S))
        z_align = _zscore(align_raw * cash_gate)

        # IPO lifecycle: favor stocks nearer to issue price
        z_ipo = _rank_norm(np.where(np.isfinite(ipo_premium) & (ipo_premium > 0.01), -np.log(ipo_premium), np.nan))

        score = (VALUE_W * z_value + QUALITY_W * z_quality + GROWTH_W * z_growth + CF_W * z_cf_premium + ALIGN_W * z_align + IPO_W * z_ipo)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
