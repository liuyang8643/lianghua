import numpy as np

MIN_PRICE = 2.0
MV_SCALE = 1e8
ROE_W = 0.35
GM_W = 0.15
OCF_W = 0.15
VALUE_W = 0.20
GROWTH_W = 0.15
SIGMOID_STEEPNESS = 0.7
DECAY_FLOOR = 0.35
TIEBREAK_EPS = 0.0003

__thesis__ = "质量与成长非线性门控的小市值溢价"


class Factor_20260605_192540_g2_2:
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        o = panel['open']
        ts = panel['total_share']
        st = panel['st_mask']

        base_valid = ~np.isnan(o) & (o >= MIN_PRICE) & ~st

        _ts = np.where(np.isfinite(ts) & (ts > 0), ts, 1.0)
        _o = np.where(np.isfinite(o) & (o > 0), o, MIN_PRICE)
        mv = _o * _ts / MV_SCALE
        log_mv = np.log(np.maximum(mv, 1e-4))

        def _imp(x):
            med = np.nanmedian(x, axis=1, keepdims=True)
            out = np.where(np.isfinite(x), x, med)
            return np.where(np.isfinite(out), out, 0.0)

        def _z(x):
            mu = np.nanmean(x, axis=1, keepdims=True)
            sd = np.nanstd(x, axis=1, keepdims=True)
            return (x - mu) / np.maximum(sd, 1e-8)

        roe_z = _z(_imp(panel['roe']))
        gm_z = _z(_imp(panel['gross_margin']))
        ocf_z = _z(_imp(panel['operating_cf_ps']))
        quality = roe_z * ROE_W + gm_z * GM_W + ocf_z * OCF_W

        eps_i = _imp(panel['eps'])
        ep = eps_i / np.maximum(_o, 1e-4)
        ep_c = np.clip(ep, -5.0, 5.0)
        value = _z(ep_c)

        py_i = _imp(panel['profit_yoy'])
        ry_i = _imp(panel['revenue_yoy'])
        growth = np.tanh(py_i * 0.01) * 0.6 + np.tanh(ry_i * 0.01) * 0.4

        comp = quality + value * VALUE_W + growth * GROWTH_W
        gate = 1.0 / (1.0 + np.exp(-comp * SIGMOID_STEEPNESS))

        score = -log_mv * (DECAY_FLOOR + gate * (1.0 - DECAY_FLOOR)) + value * TIEBREAK_EPS

        return np.where(base_valid & np.isfinite(score), score.astype(np.float64), np.nan)
