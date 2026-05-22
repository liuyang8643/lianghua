import numpy as np
from scipy.ndimage import uniform_filter1d

MIN_PRICE = 2.0


def _roll_mean(data, window):
    valid = ~np.isnan(data)
    filled = np.where(valid, data, 0.0)
    s = uniform_filter1d(filled, window, axis=0, mode='constant', cval=0.0, origin=(window - 1) // 2) * window
    c = uniform_filter1d(valid.astype(np.float64), window, axis=0, mode='constant', cval=0.0, origin=(window - 1) // 2) * window
    out = np.full_like(data, np.nan)
    ok = c > 0
    out[ok] = s[ok] / c[ok]
    return out


class ATR14:
    hist_days = 17

    def calc_batch(self, panel: dict) -> np.ndarray:
        high = panel["high"]; low = panel["low"]; close = panel["close"]
        opn = panel["open"]; st = panel["st_mask"]
        valid = ~np.isnan(opn) & (opn >= MIN_PRICE) & ~st
        tr_arr = np.full_like(close, np.nan)
        tr_arr[2:] = np.maximum.reduce([
            high[1:-1] - low[1:-1], np.abs(high[1:-1] - close[:-2]), np.abs(low[1:-1] - close[:-2]),
        ])
        atr14 = _roll_mean(tr_arr, 14)
        return np.where(valid, -atr14, np.nan)
