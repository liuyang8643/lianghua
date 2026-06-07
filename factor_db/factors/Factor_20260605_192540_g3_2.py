import numpy as np

MIN_RAW_PRICE = 2.0

__thesis__ = "盈利增长与小市值交叉融合，成长性为小盘溢价提供非线性加成"

GROWTH_AMPLIFY = 0.6
SIZE_WEIGHT = 0.35


class Factor_20260605_192540_g3_2:
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        open_ = panel['open']
        total_share = panel['total_share']
        profit_yoy = panel['profit_yoy']
        st_mask = panel['st_mask']

        base_valid = ~np.isnan(open_) & (open_ >= MIN_RAW_PRICE) & ~st_mask

        ts = total_share.astype(np.float64)
        ts_med = np.nanmedian(ts, axis=1, keepdims=True)
        ts_imp = np.where(np.isfinite(ts) & (ts > 0), ts, np.where(np.isfinite(ts_med) & (ts_med > 0), ts_med, 1e8))
        log_mcap = np.log(open_ * ts_imp)

        py = profit_yoy.astype(np.float64)
        py_med = np.nanmedian(py, axis=1, keepdims=True)
        py_imp = np.where(np.isfinite(py), py, py_med)

        def _zscore(x):
            med = np.nanmedian(x, axis=1, keepdims=True)
            mad = np.nanmedian(np.abs(x - med), axis=1, keepdims=True) * 1.4826
            return (x - med) / np.where(mad > 1e-12, mad, 1.0)

        z_growth = _zscore(py_imp)
        z_size = _zscore(-log_mcap)

        growth_gate = np.tanh(z_growth)
        score = z_size * (1.0 + GROWTH_AMPLIFY * growth_gate) + SIZE_WEIGHT * z_growth

        return np.where(base_valid & np.isfinite(score), score.astype(np.float64), np.nan)
