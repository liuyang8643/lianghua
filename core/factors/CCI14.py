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


class CCI14:
    hist_days = 17

    def calc_batch(self, panel: dict) -> np.ndarray:
        high = panel["high"]; low = panel["low"]; close = panel["close"]
        opn = panel["open"]; st = panel["st_mask"]
        valid = ~np.isnan(opn) & (opn >= MIN_PRICE) & ~st
        N, S = close.shape
        tp_arr = (high + low + close) / 3.0
        tp_sma = _roll_mean_T1(tp_arr, 14)
        abs_diff = np.abs(tp_arr - tp_sma)
        rm_abs = _roll_mean(abs_diff, 14)
        tp_mad = np.full((N, S), np.nan, dtype=np.float32)
        tp_mad[14:] = rm_abs[13:N - 1]
        tp_prev = np.roll(tp_arr, 1, axis=0); tp_prev[0] = np.nan
        cci = np.full((N, S), np.nan, dtype=np.float32)
        cci[14:] = np.where(tp_mad[14:] > 0, (tp_prev[14:] - tp_sma[14:]) / (0.015 * tp_mad[14:]), 0.0)
        return np.where(valid, cci, np.nan)
