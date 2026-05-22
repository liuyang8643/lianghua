import numpy as np

class MarginExpansion:
  """利润率扩张加速度 — (profit_yoy - revenue_yoy) 纯因子"""
  hist_days = 0

  def calc_batch(self, panel: dict) -> np.ndarray:
    raw_open = panel["open"]
    st_mask = panel["st_mask"]
    profit_yoy = panel["profit_yoy"]
    revenue_yoy = panel["revenue_yoy"]

    base_valid = (
      ~np.isnan(raw_open)
      & (raw_open >= 2.0)
      & ~st_mask
    )
    margin_expansion = profit_yoy - revenue_yoy
    return np.where(base_valid, margin_expansion, np.nan)
