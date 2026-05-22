import numpy as np
from scipy.ndimage import uniform_filter1d


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


def _roll_std_returns(daily_ret, window):
    std = _roll_std(daily_ret, window)
    N = daily_ret.shape[0]
    out = np.full_like(daily_ret, np.nan)
    out[window:] = std[window - 1:N - 1]
    return out


class TMC_ProfitYoy_25_LowVol:
    hist_days = 22

    def calc_batch(self, panel: dict) -> np.ndarray:
        raw_open = panel["open"]
        total_share = panel["total_share"]
        st_mask = panel["st_mask"]
        profit_yoy = panel["profit_yoy"]
        n_dates, n_stocks = raw_open.shape

        base_valid = (
            ~np.isnan(raw_open)
            & (raw_open >= 2.0)
            & ~np.isnan(total_share)
            & (total_share > 0)
            & ~st_mask
        )

        daily_ret = np.full((n_dates, n_stocks), np.nan, dtype=np.float32)
        daily_ret[1:] = raw_open[1:] / raw_open[:-1] - 1.0

        vol = _roll_std_returns(daily_ret, 21)

        total_mv_yi = (raw_open * total_share) / 1e8
        py_clean = np.where(base_valid & ~np.isnan(profit_yoy), profit_yoy, 0.0)
        vol_clean = np.where(base_valid & ~np.isnan(vol), vol, 0.0)

        score = -total_mv_yi + 0.25 * py_clean - 0.10 * vol_clean
        return np.where(base_valid, score, np.nan)
