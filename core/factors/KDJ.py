"""KDJ随机指标因子 — K-D 原始值."""
import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.signal import lfilter

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


class KDJ:
    hist_days = 20

    def calc_batch(self, panel: dict) -> np.ndarray:
        high = panel["high"]
        low = panel["low"]
        close = panel["close"]
        opn = panel["open"]
        st = panel["st_mask"]
        valid = ~np.isnan(opn) & (opn >= MIN_PRICE) & ~st

        cp = np.roll(close, 1, axis=0); cp[0] = np.nan
        h9 = _roll_max_T1(high, 9)
        l9 = _roll_min_T1(low, 9)
        den = h9 - l9
        rsv = np.where(den > 0, (cp - l9) / den * 100.0, 50.0)

        alpha = 1.0 / 3.0
        rsv_mask = ~np.isnan(rsv)
        clean = np.where(rsv_mask, rsv, 50.0).astype(np.float64)

        b = [alpha]
        a = [1.0, -(1.0 - alpha)]
        K_raw = lfilter(b, a, clean, axis=0)
        D_raw = lfilter(b, a, K_raw, axis=0)

        K = np.where(rsv_mask, K_raw, np.nan)
        D = np.where(rsv_mask, D_raw, np.nan)

        return np.where(valid, K - D, np.nan)
