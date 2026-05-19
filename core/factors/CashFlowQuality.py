import numpy as np
import pandas as pd

from core.factors.helpers import BaseFactor

MIN_RAW_PRICE = 2.0


class CashFlowQuality(BaseFactor):
  """现金流质量因子 — operating_cf_ps / eps，纯矩阵计算"""

  hist_days = 0

  def calc_batch(self, panel: dict) -> pd.DataFrame:
    stock_codes = panel["stock_codes"]
    trade_dates = panel["trade_dates"]
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

    cf_quality[~base_valid] = np.nan

    return pd.DataFrame(cf_quality, index=trade_dates, columns=stock_codes)
