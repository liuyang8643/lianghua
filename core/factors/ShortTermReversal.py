import numpy as np

MIN_RAW_PRICE = 2.0


class ShortTermReversal:
  """短期反转因子 — 10日 open-to-open 收益率反转，纯矩阵计算"""

  hist_days = 10

  def calc_batch(self, panel: dict) -> np.ndarray:
    trade_dates = panel["trade_dates"]
    stock_codes = panel["stock_codes"]
    open_prices = panel["open"]
    st_mask = panel["st_mask"]

    n_dates, n_stocks = len(trade_dates), len(stock_codes)
    returns = np.full((n_dates, n_stocks), np.nan, dtype=np.float32)
    returns[self.hist_days:, :] = open_prices[self.hist_days:, :] / open_prices[:-self.hist_days, :] - 1

    base_valid = (
      ~np.isnan(open_prices)
      & (open_prices >= MIN_RAW_PRICE)
      & ~st_mask
    )

    score = -returns
    return np.where(base_valid, score, np.nan)
