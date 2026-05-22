import numpy as np

MIN_PRICE = 2.0

class EPValuation:
  """盈利率估值: eps/open, 高EP=低估值=高分."""
  hist_days = 2

  def calc_batch(self, panel: dict) -> np.ndarray:
    opn = panel["open"]; st = panel["st_mask"]; eps = panel["eps"]
    valid = ~np.isnan(opn) & (opn >= MIN_PRICE) & ~st
    ep = eps / np.where(opn > 0, opn, np.nan)
    return np.where(valid, ep, np.nan)
