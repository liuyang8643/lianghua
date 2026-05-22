import numpy as np

MIN_PRICE = 2.0

class CloseMom21D:
  """21日收盘动量."""
  hist_days = 23

  def calc_batch(self, panel: dict) -> np.ndarray:
    close = panel["close"]; opn = panel["open"]; st = panel["st_mask"]
    valid = ~np.isnan(opn) & (opn >= MIN_PRICE) & ~st
    N = close.shape[0]
    mom = np.full_like(close, np.nan)
    mom[22:] = close[21:-1] / close[:-22] - 1.0
    return np.where(valid, mom, np.nan)
