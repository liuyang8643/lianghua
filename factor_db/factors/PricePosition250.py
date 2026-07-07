import numpy as np

MIN_RAW_PRICE = 2.0


def _shift(arr):
    result = np.empty_like(arr)
    result[0] = np.nan
    result[1:] = arr[:-1]
    return result


class PricePosition250:
    """250日价格位置 — close[T-1] / 250d最高close，位置越低排名越高（均值回归锚定）"""
    hist_days = 250

    def calc_batch(self, panel: dict) -> np.ndarray:
        close_known = _shift(panel["close"])
        raw_open = panel["close"]
        st_mask = panel["st_mask"]

        w = self.hist_days
        n_dates = close_known.shape[0]
        rolling_max = np.full_like(close_known, np.nan, dtype=float)

        for t in range(w, n_dates):
            window = close_known[t - w:t]
            rolling_max[t] = np.nanmax(window, axis=0)

        with np.errstate(divide='ignore', invalid='ignore'):
            prox = close_known / rolling_max

        base_valid = (
            ~np.isnan(raw_open)
            & (raw_open >= MIN_RAW_PRICE)
            & ~st_mask
            & ~np.isnan(prox)
        )
        return np.where(base_valid, -prox, np.nan)
