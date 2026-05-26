import numpy as np

MIN_RAW_PRICE = 2.0


def _shift(arr):
    result = np.empty_like(arr)
    result[0] = np.nan
    result[1:] = arr[:-1]
    return result


class AmihudIlliquidity:
  """Amihud非流动性 — |ret|/amount均值（基于已知数据），不流动溢价"""
  hist_days = 20

  def calc_batch(self, panel: dict) -> np.ndarray:
    close_known = _shift(panel["close"])
    amount_known = _shift(panel["amount"])
    raw_open = panel["open"]
    st_mask = panel["st_mask"]

    ret = np.empty_like(close_known)
    ret[0] = np.nan
    ret[1:] = close_known[1:] / close_known[:-1] - 1.0

    with np.errstate(divide='ignore', invalid='ignore'):
        illiq = np.abs(ret) / (amount_known / 1e8)

    w = self.hist_days
    n_valid = np.cumsum(~np.isnan(illiq), axis=0).astype(float)
    cum_sum = np.cumsum(np.where(np.isnan(illiq), 0.0, illiq), axis=0)

    avg_illiq = np.empty_like(illiq, dtype=float)
    avg_illiq[:w] = np.nan
    avg_illiq[w:] = (cum_sum[w:] - cum_sum[:-w]) / (n_valid[w:] - n_valid[:-w])

    base_valid = (
      ~np.isnan(raw_open)
      & (raw_open >= MIN_RAW_PRICE)
      & ~st_mask
      & ~np.isnan(avg_illiq)
    )
    return np.where(base_valid, avg_illiq, np.nan)
