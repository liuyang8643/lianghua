"""成交量放量因子 — 量比原始值."""
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


def _roll_mean_T1(data, window):
    rm = _roll_mean(data, window)
    N = data.shape[0]
    out = np.full_like(data, np.nan)
    out[window:] = rm[window - 1:N - 1]
    return out


class Turnover:
    hist_days = 22

    def calc_batch(self, panel: dict) -> np.ndarray:
        volume = panel["volume"]
        opn = panel["open"]
        st = panel["st_mask"]
        valid = ~np.isnan(opn) & (opn >= MIN_PRICE) & ~st
        N, S = volume.shape

        vol_prev = np.roll(volume, 1, axis=0); vol_prev[0] = np.nan
        ma5_t1 = _roll_mean_T1(volume, 5)
        vol_ratio = np.full_like(volume, np.nan)
        vol_ratio[6:] = vol_prev[6:] / np.where(ma5_t1[6:] > 0, ma5_t1[6:], np.nan)

        return np.where(valid, vol_ratio, np.nan)
