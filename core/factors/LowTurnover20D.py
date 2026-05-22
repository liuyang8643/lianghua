import numpy as np
from scipy.ndimage import uniform_filter1d

MIN_RAW_PRICE = 2.0


def _roll_mean(data, window):
    valid = ~np.isnan(data)
    filled = np.where(valid, data, 0.0)
    s = uniform_filter1d(filled, window, axis=0, mode='constant', cval=0.0, origin=(window - 1) // 2) * window
    c = uniform_filter1d(valid.astype(np.float64), window, axis=0, mode='constant', cval=0.0, origin=(window - 1) // 2) * window
    out = np.full_like(data, np.nan)
    ok = c > 0
    out[ok] = s[ok] / c[ok]
    return out


class LowTurnover20D:
    hist_days = 22

    def calc_batch(self, panel: dict) -> np.ndarray:
        raw_open = panel["open"]
        volume = panel["volume"]
        total_share = panel["total_share"]
        st_mask = panel["st_mask"]
        n_dates, n_stocks = volume.shape

        base_valid = (
            ~np.isnan(raw_open)
            & (raw_open >= MIN_RAW_PRICE)
            & ~np.isnan(total_share)
            & (total_share > 0)
            & ~st_mask
        )

        valid_share = total_share > 0
        daily_turnover = np.where(valid_share, volume / total_share, np.nan)
        turnover_known = np.full((n_dates, n_stocks), np.nan, dtype=np.float32)
        turnover_known[1:] = daily_turnover[:-1]

        avg_turnover_raw = _roll_mean(turnover_known, 20)
        avg_turnover = np.full((n_dates, n_stocks), np.nan, dtype=np.float32)
        avg_turnover[21:] = avg_turnover_raw[21:]

        return np.where(base_valid, -avg_turnover, np.nan)
