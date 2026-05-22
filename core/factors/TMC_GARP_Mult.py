import numpy as np

class TMC_GARP_Mult:
  """乘法合成GARP: size * growth * value * quality 非线性交互."""

  hist_days = 0

  def calc_batch(self, panel: dict) -> np.ndarray:
    raw_open = panel["open"]
    total_share = panel["total_share"]
    st_mask = panel["st_mask"]
    profit_yoy = panel["profit_yoy"]
    eps = panel["eps"]
    operating_cf_ps = panel["operating_cf_ps"]

    base_valid = (
      ~np.isnan(raw_open) & (raw_open >= 2.0)
      & ~np.isnan(total_share) & (total_share > 0)
      & ~st_mask
    )

    total_mv_yi = (raw_open * total_share) / 1e8
    size_val = -total_mv_yi

    py_contrib = np.where(base_valid & ~np.isnan(profit_yoy), 1 + 0.30 * profit_yoy, 1.0)
    ep_arr = eps / np.where(raw_open > 0, raw_open, np.nan)
    ep_contrib = np.where(base_valid & ~np.isnan(ep_arr), 1 + 0.15 * ep_arr, 1.0)
    cfq_contrib = np.where(base_valid & ~np.isnan(operating_cf_ps), 1 + 0.10 * operating_cf_ps, 1.0)

    score = size_val * py_contrib * ep_contrib * cfq_contrib
    return np.where(base_valid, score, np.nan)
