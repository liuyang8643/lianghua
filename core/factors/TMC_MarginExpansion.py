import numpy as np

class SmallCapMarginExpansion:
  """小盘 + 利润率扩张（profit_yoy - revenue_yoy）— 利润增速超越营收增速"""
  hist_days = 0

  def calc_batch(self, panel: dict) -> np.ndarray:
    raw_open = panel["open"]
    total_share = panel["total_share"]
    st_mask = panel["st_mask"]
    profit_yoy = panel["profit_yoy"]
    revenue_yoy = panel["revenue_yoy"]

    base_valid = (
      ~np.isnan(raw_open)
      & (raw_open >= 2.0)
      & ~np.isnan(total_share)
      & (total_share > 0)
      & ~st_mask
    )

    total_mv_yi = (raw_open * total_share) / 1e8
    margin_expansion = profit_yoy - revenue_yoy
    me_clean = np.where(base_valid & ~np.isnan(margin_expansion), margin_expansion, 0.0)
    py_clean = np.where(base_valid & ~np.isnan(profit_yoy), profit_yoy, 0.0)

    score = -total_mv_yi + 0.15 * me_clean + 0.10 * py_clean
    return np.where(base_valid, score, np.nan)
