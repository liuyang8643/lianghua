import numpy as np

MIN_RAW_PRICE = 2.0


def _shift(arr):
    result = np.empty_like(arr)
    result[0] = np.nan
    result[1:] = arr[:-1]
    return result


class MAX5:
  """彩票偏好 — 过去N日最大日收益（基于已知收盘），高MAX=彩票股=低收益"""
  hist_days = 21

  def calc_batch(self, panel: dict) -> np.ndarray:
    close_known = _shift(panel["close"])
    raw_open = panel["open"]
    st_mask = panel["st_mask"]

    ret = np.empty_like(close_known)
    ret[0] = np.nan
    ret[1:] = close_known[1:] / close_known[:-1] - 1.0

    n_dates = ret.shape[0]
    max_ret = np.full_like(ret, np.nan, dtype=float)
    w = self.hist_days

    for t in range(w, n_dates):
      window = ret[t - w:t]
      max_ret[t] = np.nanmax(window, axis=0)

    base_valid = (
      ~np.isnan(raw_open)
      & (raw_open >= MIN_RAW_PRICE)
      & ~st_mask
      & ~np.isnan(max_ret)
    )
    return np.where(base_valid, -max_ret, np.nan)
