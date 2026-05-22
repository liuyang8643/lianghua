import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d

MIN_PRICE = 2.0


def _roll_max_T1(data, window):
    N = data.shape[0]
    out = np.full_like(data, np.nan)
    out[window:] = maximum_filter1d(
        data, window, axis=0, mode='constant', cval=-np.inf, origin=(window - 1) // 2
    )[window - 1:N - 1]
    return out


def _roll_min_T1(data, window):
    N = data.shape[0]
    out = np.full_like(data, np.nan)
    out[window:] = minimum_filter1d(
        data, window, axis=0, mode='constant', cval=np.inf, origin=(window - 1) // 2
    )[window - 1:N - 1]
    return out


class PricePosition256D:
    hist_days = 258

    def calc_batch(self, panel: dict) -> np.ndarray:
        high = panel["high"]; low = panel["low"]; close = panel["close"]
        opn = panel["open"]; st = panel["st_mask"]
        valid = ~np.isnan(opn) & (opn >= MIN_PRICE) & ~st
        cp = np.roll(close, 1, axis=0); cp[0] = np.nan
        h256 = _roll_max_T1(high, 256); l256 = _roll_min_T1(low, 256)
        den = h256 - l256
        pos = np.where(den > 0, (cp - l256) / den, 0.5)
        return np.where(valid, pos, np.nan)
