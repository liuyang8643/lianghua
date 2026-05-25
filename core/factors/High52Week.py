import numpy as np

MIN_RAW_PRICE = 2.0


def _shift(arr):
    result = np.empty_like(arr)
    result[0] = np.nan
    result[1:] = arr[:-1]
    return result


class High52Week:
  """52周高点锚定 — close/252d最高价，远离高点的股票被低估（锚定偏差）"""
  hist_days = 252

  def calc_batch(self, panel: dict) -> np.ndarray:
    close_known = _shift(panel["close"])
    raw_open = panel["open"]
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
