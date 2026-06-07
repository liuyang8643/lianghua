import numpy as np

VALUE_W = 0.24
QUALITY_W = 0.22
GROWTH_W = 0.18
OP_EFF_W = 0.12
SYNERGY_W = 0.14

CRED_S = 2.0
CASH_S = 2.5
ROE_S = 2.0
ACCRUAL_S = 2.0
TRUST_S = 2.0
GROWTH_S = 1.8
GM_S = 2.0
EFF_S = 2.0
CONVEX_S = 1.5

__thesis__ = "可信度加权应计修正价值与信任质量凸性增长协同"


class Factor_20260531_005210_g29_1:
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
            growth_eff = panel['profit_yoy'] / np.where(np.isfinite(panel['revenue_yoy']) & (np.abs(panel['revenue_yoy']) > 1e-8), np.abs(panel['revenue_yoy']), np.nan)

        roe_s = np.sign(panel['roe']) * (np.abs(panel['roe']) ** (1/3))
        gm_s = np.sign(panel['gross_margin']) * (np.abs(panel['gross_margin']) ** (1/3))

        z_roe = _zscore(roe_s)
        z_gm = _zscore(gm_s)
        z_ey = _zscore(ey)
        z_cf = _zscore(cf_yield)
        z_cash = _zscore(cash_conf)
        z_profit = _zscore(panel['profit_yoy'])
        z_rev = _zscore(panel['revenue_yoy'])
        z_accruals = _zscore(accruals)

        # 三元可信度门: roe × gm × cash_conf 三因子交叉验证
        cred_raw = z_roe * z_gm * z_cash
        cred_gate = 0.5 + 0.5 * np.tanh(cred_raw * CRED_S)

        cash_gate = 0.5 + 0.5 * np.tanh(z_cash * CASH_S)
        roe_gate = 0.5 + 0.5 * np.tanh(z_roe * ROE_S)

        # 价值端: 应计修正 EY × 三元可信度门 × 凸性放大器
        accrual_mod = 0.5 + 0.5 * np.tanh(z_accruals * ACCRUAL_S)
        convex_amp = np.sqrt(np.where(np.isfinite(z_ey ** 2 + z_cf ** 2) & (z_ey ** 2 + z_cf ** 2 > 1e-12), z_ey ** 2 + z_cf ** 2, np.nan))
        value_raw = z_ey * accrual_mod * cred_gate * np.tanh(convex_amp * CONVEX_S)
        z_value = _zscore(value_raw + 0.03 * _rank_norm(np.where(np.isfinite(panel['open']) & (panel['open'] > 0.01), np.log(panel['open']), np.nan)))

        # 质量端: 信任桥接 = 现金质量 × 盈利质量 × 共识配对
        earn_quality = z_roe + z_gm
        cash_quality = z_cf + z_cash
        trust_raw = cash_quality * earn_quality
        trust_gate = 0.5 + 0.5 * np.tanh(trust_raw * TRUST_S)
        t_roe = np.tanh(z_roe)
        t_gm = np.tanh(z_gm)
        t_cash = np.tanh(z_cash)
        consensus_pairwise = (t_roe * t_gm + t_roe * t_cash + t_gm * t_cash) / 3.0
        quality_raw = np.tanh(trust_raw * TRUST_S) * consensus_pairwise * cash_gate * roe_gate
        z_quality = _zscore(_zscore(quality_raw) + 0.03 * _rank_norm(np.where(np.isfinite(panel['open'] * panel['total_share']) & (panel['open'] * panel['total_share'] > 1.0), np.log(panel['open'] * panel['total_share']), np.nan)))

        # 增长端: 双引擎（有机+财务）× 增长效率门
        g_gate = 0.5 + 0.5 * np.tanh(growth_eff * GROWTH_S)
        gm_gate = 0.5 + 0.5 * np.tanh(z_gm * GM_S)
        z_organic = _zscore(z_rev * gm_gate * cash_gate)
        z_financial = _zscore(z_profit * z_cash * roe_gate)
        z_growth = _zscore((z_organic + z_financial) * g_gate * trust_gate)

        # 经营效率桥: gm × cf 交叉验证
        op_eff = np.tanh(z_gm * z_cf * EFF_S)
        z_op_eff = _zscore(op_eff * cred_gate)

        # 协同: 质量 × 价值交互项
        z_synergy = _zscore(z_quality * z_value)

        # 规模惩罚

        score = (VALUE_W * z_value + QUALITY_W * z_quality + GROWTH_W * z_growth + OP_EFF_W * z_op_eff + SYNERGY_W * z_synergy)

        return np.where(base_valid & np.isfinite(score), score, np.nan)
