import numpy as np
from scipy.signal import lfilter

MIN_PRICE = 2.0

class EWMADivergence:
  hist_days = 4

  def calc_batch(self, panel: dict) -> np.ndarray:
    close = panel["close"]; opn = panel["open"]; st = panel["st_mask"]
    valid = ~np.isnan(opn) & (opn >= MIN_PRICE) & ~st
    alpha = 0.15
    b = [alpha]; a = [1.0, -(1.0 - alpha)]
    filled = np.nan_to_num(close, nan=0.0)
    ewma = lfilter(b, a, filled, axis=0).astype(np.float32)
    div = np.empty_like(close, dtype=np.float32)
    div[0] = np.nan
    div[1:] = close[:-1] / ewma[:-1] - np.float32(1.0)
    return np.where(valid, div, np.nan)
