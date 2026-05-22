import numpy as np

MIN_RAW_PRICE = 2.0


class DailyRange:
  """日内振幅 — (high-low)/close均值，高振幅=分歧大=负面"""
  hist_days = 20

  def calc_batch(self, panel: dict) -> np.ndarray:
    high = panel["high"]
    low = panel["low"]
    close = panel["close"]
    raw_open = panel["open"]
    st_mask = panel["st_mask"]

    with np.errstate(divide='ignore', invalid='ignore'):
        day_range = (high - low) / close

    w = self.hist_days
    n_valid = np.cumsum(~np.isnan(day_range), axis=0).astype(float)
    cum_sum = np.cumsum(np.where(np.isnan(day_range), 0.0, day_range), axis=0)

    avg_range = np.empty_like(day_range, dtype=float)
    avg_range[:w] = np.nan
    avg_range[w:] = (cum_sum[w:] - cum_sum[:-w]) / (n_valid[w:] - n_valid[:-w])

    base_valid = (
      ~np.isnan(raw_open)
      & (raw_open >= MIN_RAW_PRICE)
      & ~st_mask
      & ~np.isnan(avg_range)
    )
    return np.where(base_valid, -avg_range, np.nan)
