import numpy as np

MIN_PRICE = 2.0

class ProfitYoy:
  """纯利润增速因子(原始值)."""
  hist_days = 2

  def calc_batch(self, panel: dict) -> np.ndarray:
    opn = panel["open"]; st = panel["st_mask"]; profit_yoy = panel["profit_yoy"]
    valid = ~np.isnan(opn) & (opn >= MIN_PRICE) & ~st
    return np.where(valid, profit_yoy, np.nan)
