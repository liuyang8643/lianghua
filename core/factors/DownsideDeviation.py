import numpy as np

MIN_RAW_PRICE = 2.0


def _shift(arr):
    result = np.empty_like(arr)
    result[0] = np.nan
    result[1:] = arr[:-1]
    return result


class DownsideDeviation:
  """下行波动 — 已知负收益的标准差，高下行风险=低未来收益"""
  hist_days = 60

  def calc_batch(self, panel: dict) -> np.ndarray:
    close_known = _shift(panel["close"])
    raw_open = panel["open"]
    st_mask = panel["st_mask"]

    ret = np.empty_like(close_known)
    ret[0] = np.nan
    ret[1:] = close_known[1:] / close_known[:-1] - 1.0

    neg_ret = np.minimum(ret, 0.0)
    neg_ret_sq = neg_ret * neg_ret

    n_valid = np.cumsum(~np.isnan(neg_ret_sq), axis=0).astype(float)
    cum_sum = np.cumsum(np.where(np.isnan(neg_ret_sq), 0.0, neg_ret_sq), axis=0)

    w = self.hist_days
    dd = np.empty_like(ret, dtype=float)
    dd[:w] = np.nan
    dd[w:] = np.sqrt((cum_sum[w:] - cum_sum[:-w]) / (n_valid[w:] - n_valid[:-w]))

    base_valid = (
      ~np.isnan(raw_open)
      & (raw_open >= MIN_RAW_PRICE)
      & ~st_mask
      & ~np.isnan(dd)
    )
    return np.where(base_valid, -dd, np.nan)
