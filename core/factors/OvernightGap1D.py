import numpy as np

MIN_PRICE = 2.0

class OvernightGap1D:
  """单日隔夜跳空: open[t]/close[t-1] - 1."""
  hist_days = 3

  def calc_batch(self, panel: dict) -> np.ndarray:
    opn = panel["open"]; close = panel["close"]; st = panel["st_mask"]
    valid = ~np.isnan(opn) & (opn >= MIN_PRICE) & ~st
    N = opn.shape[0]
    gap = np.full_like(opn, np.nan)
    gap[1:] = opn[1:] / close[:-1] - 1.0
    return np.where(valid, gap, np.nan)
