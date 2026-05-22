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


def _roll_std(data, window):
    valid = ~np.isnan(data)
    filled = np.where(valid, data, 0.0)
    filled2 = np.where(valid, data * data, 0.0)
    s = uniform_filter1d(filled, window, axis=0, mode='constant', cval=0.0, origin=(window - 1) // 2) * window
    s2 = uniform_filter1d(filled2, window, axis=0, mode='constant', cval=0.0, origin=(window - 1) // 2) * window
    c = uniform_filter1d(valid.astype(np.float64), window, axis=0, mode='constant', cval=0.0, origin=(window - 1) // 2) * window
    out = np.full_like(data, np.nan)
    ok = c > 0
    mv = s[ok] / c[ok]
    out[ok] = np.sqrt(np.maximum(0, s2[ok] / c[ok] - mv * mv))
    return out


def _roll_mean_T1(data, window):
    rm = _roll_mean(data, window)
    N = data.shape[0]
    out = np.full_like(data, np.nan)
    out[window:] = rm[window - 1:N - 1]
    return out


def _roll_std_T1(data, window):
    std = _roll_std(data, window)
    N = data.shape[0]
    out = np.full_like(data, np.nan)
    out[window:] = std[window - 1:N - 1]
    return out


class BB20Position:
    hist_days = 22

    def calc_batch(self, panel: dict) -> np.ndarray:
        close = panel["close"]; opn = panel["open"]; st = panel["st_mask"]
        valid = ~np.isnan(opn) & (opn >= MIN_PRICE) & ~st
        cp = np.roll(close, 1, axis=0); cp[0] = np.nan
        ma20 = _roll_mean_T1(close, 20); std20 = _roll_std_T1(close, 20)
        upper = ma20 + 2.0 * std20; lower = ma20 - 2.0 * std20
        den = upper - lower
        pos = np.where(den > 0, (cp - lower) / den, 0.5)
        return np.where(valid, pos, np.nan)
