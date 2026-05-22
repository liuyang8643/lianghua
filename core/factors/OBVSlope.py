import numpy as np

MIN_PRICE = 2.0

class OBVSlope:
  """OBV斜率: (OBV[t-1]-OBV[t-10]) / |OBV[t-10]|."""
  hist_days = 13

  def calc_batch(self, panel: dict) -> np.ndarray:
    close = panel["close"]; vol = panel["volume"]
    opn = panel["open"]; st = panel["st_mask"]
    valid = ~np.isnan(opn) & (opn >= MIN_PRICE) & ~st
    N, S = close.shape
    cp2 = np.roll(close, 1, axis=0); cp2[0] = np.nan
    cp3 = np.roll(close, 2, axis=0); cp3[:2] = np.nan
    vp = np.roll(vol, 1, axis=0); vp[0] = np.nan
    obv_inc = np.where(cp2 > cp3, vp, np.where(cp2 < cp3, -vp, 0.0))
    obv = np.cumsum(np.nan_to_num(obv_inc, nan=0.0), axis=0)
    obv_prev = np.roll(obv, 1, axis=0); obv_prev[0] = np.nan
    obv_p10 = np.roll(obv, 10, axis=0); obv_p10[:10] = np.nan
    den = np.where(np.abs(obv_p10) > 0, np.abs(obv_p10), 1.0)
    slope = np.full((N, S), np.nan, dtype=np.float32)
    slope[10:] = (obv_prev[10:] - obv_p10[10:]) / den[10:]
    return np.where(valid, slope, np.nan)
