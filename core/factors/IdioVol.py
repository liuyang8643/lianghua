import numpy as np

MIN_RAW_PRICE = 2.0


def _shift(arr):
    """将数组后移1天: result[t] = arr[t-1], result[0] = NaN"""
    result = np.empty_like(arr)
    result[0] = np.nan
    result[1:] = arr[:-1]
    return result


class IdioVol:
  """特质波动率 — 日收益率滚动标准差（基于已知收盘价），低波动=高收益"""
  hist_days = 60

  def calc_batch(self, panel: dict) -> np.ndarray:
    close_known = _shift(panel["close"])
    raw_open = panel["open"]
    st_mask = panel["st_mask"]

    ret = np.empty_like(close_known)
    ret[0] = np.nan
    ret[1:] = close_known[1:] / close_known[:-1] - 1.0

    n_valid = np.cumsum(~np.isnan(ret), axis=0).astype(float)
    cum_sum = np.cumsum(np.where(np.isnan(ret), 0.0, ret), axis=0)
    cum_sum2 = np.cumsum(np.where(np.isnan(ret), 0.0, ret * ret), axis=0)

    w = self.hist_days
    vol = np.empty_like(ret, dtype=float)
    vol[:w] = np.nan
    count = n_valid[w:] - n_valid[:-w]
    mean = (cum_sum[w:] - cum_sum[:-w]) / count
    mean_sq = (cum_sum2[w:] - cum_sum2[:-w]) / count
    var = mean_sq - mean * mean
    vol[w:] = np.sqrt(np.maximum(var, 0.0))

    base_valid = (
      ~np.isnan(raw_open)
      & (raw_open >= MIN_RAW_PRICE)
      & ~st_mask
      & ~np.isnan(vol)
    )
    return np.where(base_valid, -vol, np.nan)
