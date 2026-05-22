import numpy as np
from core.factors.helpers import BaseFactor
from core.factors.helpers.rolling import roll_max_T1, roll_min_T1

MIN_PRICE = 2.0

class WillR14(BaseFactor):
  """14日威廉指标: 低值=超卖."""
  hist_days = 16

  def calc_batch(self, panel: dict) -> np.ndarray:
    high = panel["high"]; low = panel["low"]; close = panel["close"]
    opn = panel["open"]; st = panel["st_mask"]
    valid = ~np.isnan(opn) & (opn >= MIN_PRICE) & ~st
    cp = np.roll(close, 1, axis=0); cp[0] = np.nan
    hh14 = roll_max_T1(high, 14); ll14 = roll_min_T1(low, 14)
    den = hh14 - ll14
    willr = np.where(den > 0, (hh14 - cp) / den * -100.0, -50.0)
    return np.where(valid, -willr, np.nan)
