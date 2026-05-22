import numpy as np

MIN_RAW_PRICE = 2.0


class ProfitStability:
  """盈利稳定性 — profit_yoy滚动标准差取负，盈利波动小=溢价"""
  hist_days = 4  # 4个季报期

  def calc_batch(self, panel: dict) -> np.ndarray:
    profit_yoy = panel["profit_yoy"]
    raw_open = panel["open"]
    st_mask = panel["st_mask"]

    w = self.hist_days
    n_valid = np.cumsum(~np.isnan(profit_yoy), axis=0).astype(float)
    cum_sum = np.cumsum(np.where(np.isnan(profit_yoy), 0.0, profit_yoy), axis=0)
    cum_sum2 = np.cumsum(np.where(np.isnan(profit_yoy), 0.0, profit_yoy * profit_yoy), axis=0)

    stability = np.empty_like(profit_yoy, dtype=float)
    stability[:w] = np.nan
    count = n_valid[w:] - n_valid[:-w]
    mean = (cum_sum[w:] - cum_sum[:-w]) / count
    mean_sq = (cum_sum2[w:] - cum_sum2[:-w]) / count
    var = mean_sq - mean * mean
    stability[w:] = np.sqrt(np.maximum(var, 0.0))

    base_valid = (
      ~np.isnan(raw_open)
      & (raw_open >= MIN_RAW_PRICE)
      & ~st_mask
      & ~np.isnan(stability)
    )
    return np.where(base_valid, -stability, np.nan)
