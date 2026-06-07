import numpy as np

MIN_RAW_PRICE = 2.0


def _shift(arr):
    result = np.empty_like(arr)
    result[0] = np.nan
    result[1:] = arr[:-1]
    return result


class OvernightGap:
  """隔夜跳空 — 过去N日平均隔夜收益(open/prev_close-1)，负跳空=散户恐慌=买入机会"""
  hist_days = 1

  def calc_batch(self, panel: dict) -> np.ndarray:
    open_p = panel["open"]       # 已知：开盘价
    close_prev = _shift(panel["close"])  # 前一日收盘价
    st_mask = panel["st_mask"]

    with np.errstate(divide='ignore', invalid='ignore'):
        overnight = open_p / close_prev - 1.0

    w = self.hist_days
    n_valid = np.cumsum(~np.isnan(overnight), axis=0).astype(float)
    cum_sum = np.cumsum(np.where(np.isnan(overnight), 0.0, overnight), axis=0)

    avg_overnight = np.empty_like(overnight, dtype=float)
    avg_overnight[:w] = np.nan
    avg_overnight[w:] = (cum_sum[w:] - cum_sum[:-w]) / (n_valid[w:] - n_valid[:-w])

    base_valid = (
      ~np.isnan(open_p)
      & (open_p >= MIN_RAW_PRICE)
      & ~st_mask
      & ~np.isnan(avg_overnight)
    )
    return np.where(base_valid, -avg_overnight, np.nan)
