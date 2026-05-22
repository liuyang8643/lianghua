import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d

MIN_PRICE = 2.0

class Aroon14:
    hist_days = 17

    def calc_batch(self, panel: dict) -> np.ndarray:
        high = panel["high"]; low = panel["low"]
        opn = panel["open"]; st = panel["st_mask"]
        valid = ~np.isnan(opn) & (opn >= MIN_PRICE) & ~st
        N, S = high.shape
        W = 14

        roll_max = maximum_filter1d(high, W, axis=0, mode='constant', cval=-np.inf, origin=(W - 1) // 2)
        roll_min = minimum_filter1d(low, W, axis=0, mode='constant', cval=np.inf, origin=(W - 1) // 2)

        is_new_high = np.zeros((N, S), dtype=bool)
        is_new_high[1:] = (high[1:] >= roll_max[1:]) & np.isfinite(roll_max[1:])
        is_new_low = np.zeros((N, S), dtype=bool)
        is_new_low[1:] = (low[1:] <= roll_min[1:]) & np.isfinite(roll_min[1:])

        idx = np.arange(N, dtype=np.int32)[:, None]
        last_high = np.maximum.accumulate(np.where(is_new_high, idx, 0), axis=0)
        last_low = np.maximum.accumulate(np.where(is_new_low, idx, 0), axis=0)
        days_high = np.minimum(idx - last_high, W - 1)
        days_low = np.minimum(idx - last_low, W - 1)

        aroon = ((W - days_high.astype(float)) - (W - days_low.astype(float))) / (W - 1) * 100.0
        aroon[:W] = np.nan
        aroon = np.roll(aroon, 1, axis=0); aroon[0] = np.nan
        return np.where(valid, aroon, np.nan)
