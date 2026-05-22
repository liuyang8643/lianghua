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


class ADX14Trend:
    hist_days = 17

    def calc_batch(self, panel: dict) -> np.ndarray:
        high = panel["high"]; low = panel["low"]; close = panel["close"]
        opn = panel["open"]; st = panel["st_mask"]
        valid = ~np.isnan(opn) & (opn >= MIN_PRICE) & ~st
        N, S = close.shape

        tr_arr = np.zeros((N, S))
        tr_arr[2:] = np.maximum.reduce([
            high[1:-1] - low[1:-1], np.abs(high[1:-1] - close[:-2]), np.abs(low[1:-1] - close[:-2]),
        ])
        hh = high[1:-1] - high[:-2]; ll = low[:-2] - low[1:-1]
        pdm = np.zeros((N, S)); ndm = np.zeros((N, S))
        pdm[2:] = np.where((hh > ll) & (hh > 0), hh, 0)
        ndm[2:] = np.where((ll > hh) & (ll > 0), ll, 0)

        atr14 = _roll_mean(tr_arr, 14)
        pdi14 = _roll_mean(pdm, 14) * 100.0 / np.where(atr14 > 0, atr14, np.nan)
        ndi14 = _roll_mean(ndm, 14) * 100.0 / np.where(atr14 > 0, atr14, np.nan)
        denom = pdi14 + ndi14
        dx = np.zeros((N, S))
        vv = denom > 0; dx[vv] = np.abs(pdi14[vv] - ndi14[vv]) / denom[vv] * 100.0
        adx = _roll_mean(dx, 14)
        return np.where(valid, adx, np.nan)
