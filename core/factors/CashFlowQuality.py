import numpy as np

MIN_RAW_PRICE = 2.0


class CashFlowQuality:
  """现金流质量因子 — operating_cf_ps / eps，纯矩阵计算"""

  hist_days = 0

  def calc_batch(self, panel: dict) -> np.ndarray:
    open_prices = panel["open"]
    st_mask = panel["st_mask"]
    operating_cf_ps = panel["operating_cf_ps"]
    eps = panel["eps"]

    with np.errstate(divide='ignore', invalid='ignore'):
      cf_quality = operating_cf_ps / eps

    base_valid = (
      ~np.isnan(open_prices)
      & (open_prices >= MIN_RAW_PRICE)
      & ~st_mask
      & ~np.isnan(cf_quality)
      & (eps > 0)
      & (operating_cf_ps > 0)
    )
    return np.where(base_valid, cf_quality, np.nan)
