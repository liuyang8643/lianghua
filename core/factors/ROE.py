import numpy as np

MIN_PRICE = 2.0

class ROE:
  """纯ROE因子(原始值)."""
  hist_days = 2

  def calc_batch(self, panel: dict) -> np.ndarray:
    opn = panel["open"]; st = panel["st_mask"]; roe = panel["roe"]
    valid = ~np.isnan(opn) & (opn >= MIN_PRICE) & ~st
    return np.where(valid, roe, np.nan)
