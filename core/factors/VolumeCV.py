import numpy as np

MIN_RAW_PRICE = 2.0


def _shift(arr):
    result = np.empty_like(arr)
    result[0] = np.nan
    result[1:] = arr[:-1]
    return result


class VolumeCV:
  """成交量变异系数 — std(volume)/mean(volume)(已知量)，高CV=游资炒作=负面"""
  hist_days = 20

  def calc_batch(self, panel: dict) -> np.ndarray:
    volume_known = _shift(panel["volume"])
    raw_open = panel["open"]
    st_mask = panel["st_mask"]

    w = self.hist_days
    n_valid = np.cumsum(~np.isnan(volume_known), axis=0).astype(float)
    cum_sum = np.cumsum(np.where(np.isnan(volume_known), 0.0, volume_known), axis=0)
    cum_sum2 = np.cumsum(np.where(np.isnan(volume_known), 0.0, volume_known * volume_known), axis=0)

    cv = np.empty_like(volume_known, dtype=float)
    cv[:w] = np.nan
    count = n_valid[w:] - n_valid[:-w]
    mean = (cum_sum[w:] - cum_sum[:-w]) / count
    mean_sq = (cum_sum2[w:] - cum_sum2[:-w]) / count
    var = mean_sq - mean * mean
    with np.errstate(divide='ignore', invalid='ignore'):
        cv[w:] = np.sqrt(np.maximum(var, 0.0)) / mean

    base_valid = (
      ~np.isnan(raw_open)
      & (raw_open >= MIN_RAW_PRICE)
      & ~st_mask
      & ~np.isnan(cv)
    )
    return np.where(base_valid, -cv, np.nan)
