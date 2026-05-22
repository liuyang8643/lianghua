import numpy as np
from core.factors.helpers import BaseFactor
from core.factors.helpers.rolling import roll_mean_T1

MIN_PRICE = 2.0

class RSI14Reversal(BaseFactor):
  """14日RSI: 低值=超卖反弹机会."""
  hist_days = 16

  def calc_batch(self, panel: dict) -> np.ndarray:
    close = panel["close"]; opn = panel["open"]; st = panel["st_mask"]
    valid = ~np.isnan(opn) & (opn >= MIN_PRICE) & ~st
    N, S = close.shape
    chg = np.full((N, S), np.nan); chg[1:] = close[1:] / close[:-1] - 1.0
    avg_g = roll_mean_T1(np.maximum(chg, 0), 14)
    avg_l = roll_mean_T1(np.abs(np.minimum(chg, 0)), 14)
    rs = np.full((N, S), 0.0)
    v = avg_l > 0; rs[v] = avg_g[v] / avg_l[v]
    rs[(~v) & (avg_g > 0)] = 100.0
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return np.where(valid, -rsi, np.nan)
