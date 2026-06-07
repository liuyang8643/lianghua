import numpy as np

GROWTH_W = 0.30
QUALITY_W = 0.25
SIZE_W = 0.20
REVENUE_W = 0.10
INTERACT_W = 0.10
EPS_YIELD_W = 0.05
MIN_RAW_PRICE = 2.0

__thesis__ = "利润增速与ROE质量交叉共振，叠加小盘溢价与营收验证，三维捕捉戴维斯双击"


class Factor_20260605_192540_g2_0:
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        op = panel['open']
        n_stocks = op.shape[1]
        base_valid = ~np.isnan(op) & (op >= MIN_RAW_PRICE) & ~panel['st_mask']

        def _z(x):
            xf = np.where(np.isfinite(x), x, 0.0)
            mu = xf.mean(axis=1, keepdims=True)
            sd = xf.std(axis=1, keepdims=True)
            sd = np.maximum(sd, 1e-8)
            return (xf - mu) / sd

        mv = op * panel['total_share']
        mv_ok = np.isfinite(mv) & (mv > 0)
        size_raw = -np.log(np.maximum(np.where(mv_ok, mv, 1e8), 1e6))

        py = panel['profit_yoy'].astype(np.float64)
        growth_raw = np.tanh(py * 0.4)

        roe = panel['roe'].astype(np.float64)
        quality_raw = np.tanh(roe * 0.12)

        ry = panel['revenue_yoy'].astype(np.float64)
        rev_raw = np.tanh(ry * 0.25)

        eps = panel['eps'].astype(np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            ep = eps / op
        ep_raw = np.tanh(np.where(np.isfinite(ep), ep, 0.0) * 15.0)

        growth_z = _z(growth_raw)
        quality_z = _z(quality_raw)

        score = (
            GROWTH_W * growth_z
            + QUALITY_W * quality_z
            + SIZE_W * _z(size_raw)
            + REVENUE_W * _z(rev_raw)
            + INTERACT_W * _z(growth_z * quality_z)
            + EPS_YIELD_W * _z(ep_raw)
        )

        jitter = op * np.arange(n_stocks).astype(np.float64)[None, :] * 1e-12
        score = score + jitter

        return np.where(base_valid & np.isfinite(score), score, np.nan)
