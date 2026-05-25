import numpy as np

MIN_RAW_PRICE = 2.0


def _shift(arr):
    result = np.empty_like(arr)
    result[0] = np.nan
    result[1:] = arr[:-1]
    return result


class TurnoverMean:
  """低换手率 — 日均已知换手，低换手=被忽视股=溢价"""
  hist_days = 20

  def calc_batch(self, panel: dict) -> np.ndarray:
    volume_known = _shift(panel["volume"])
    total_share = panel["total_share"]
    raw_open = panel["open"]
    st_mask = panel["st_mask"]

    with np.errstate(divide='ignore', invalid='ignore'):
        turnover = volume_known / total_share

    w = self.hist_days
    n_valid = np.cumsum(~np.isnan(turnover), axis=0).astype(float)
    cum_sum = np.cumsum(np.where(np.isnan(turnover), 0.0, turnover), axis=0)

    avg_turnover = np.empty_like(turnover, dtype=float)
    avg_turnover[:w] = np.nan
    avg_turnover[w:] = (cum_sum[w:] - cum_sum[:-w]) / (n_valid[w:] - n_valid[:-w])

    base_valid = (
      ~np.isnan(raw_open)
      & (raw_open >= MIN_RAW_PRICE)
      & ~st_mask
      & ~np.isnan(avg_turnover)
    )
    return np.where(base_valid, -avg_turnover, np.nan)
