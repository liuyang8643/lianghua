import numpy as np
import pandas as pd

from core.factors.helpers import BaseFactor

MIN_RAW_PRICE = 2.0


class SmallCap(BaseFactor):
  """小盘股因子 - 基于成交额近似市值"""

  hist_days = 60

  def calc_batch(self, panel: dict) -> pd.DataFrame:
    stock_codes = panel["stock_codes"]
    trade_dates = panel["trade_dates"]
    close = panel["open"]
    amount = panel["amount"]
    st_mask = panel["st_mask"]

    amount_df = pd.DataFrame(amount, index=trade_dates, columns=stock_codes)
    avg_amounts = amount_df.rolling(window=self.hist_days, min_periods=1).mean().values / 1e8

    base_valid = (
      ~np.isnan(close)
      & (close >= MIN_RAW_PRICE)
      & ~st_mask
      & ~np.isnan(avg_amounts)
    )

    score = 100 * np.exp(-(avg_amounts / 5))
    score[~base_valid] = np.nan

    return pd.DataFrame(score, index=trade_dates, columns=stock_codes)
