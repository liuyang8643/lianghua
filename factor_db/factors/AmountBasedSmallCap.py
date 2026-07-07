import numpy as np

MIN_RAW_PRICE = 2.0


class AmountBasedSmallCap:
  """小盘股因子 - 基于成交额近似市值"""

  hist_days = 60

  def calc_batch(self, panel: dict) -> np.ndarray:
    close = panel["close"]
    amount = panel["amount"]
    st_mask = panel["st_mask"]

    amount_known = np.empty_like(amount)
    amount_known[0] = np.nan
    amount_known[1:] = amount[:-1]

    amount_filled = np.where(np.isnan(amount_known), 0.0, amount_known)
    cum_amount = np.cumsum(amount_filled, axis=0)
    cum_count = np.cumsum(~np.isnan(amount_known), axis=0).astype(float)
    w = self.hist_days
    avg_amounts = np.empty_like(amount, dtype=float)
    avg_amounts[:w] = cum_amount[:w] / cum_count[:w]
    avg_amounts[w:] = (cum_amount[w:] - cum_amount[:-w]) / (cum_count[w:] - cum_count[:-w])
    avg_amounts /= 1e8

    base_valid = (
      ~np.isnan(close)
      & (close >= MIN_RAW_PRICE)
      & ~st_mask
      & ~np.isnan(avg_amounts)
    )

    score = 100 * np.exp(-(avg_amounts / 5))
    return np.where(base_valid, score, np.nan)
