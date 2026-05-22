import numpy as np

MIN_RAW_PRICE = 2.0


class ShortTermReversal:
  """短期反转 — A股无动量有反转，近期跌的股票未来涨"""
  hist_days = 10

  def calc_batch(self, panel: dict) -> np.ndarray:
    close = panel["close"]
    raw_open = panel["open"]
    st_mask = panel["st_mask"]

    w = self.hist_days
    reversal = np.empty_like(close)
    reversal[:w] = np.nan
    reversal[w:] = -(close[w:] / close[:-w] - 1.0)

    base_valid = (
      ~np.isnan(raw_open)
      & (raw_open >= MIN_RAW_PRICE)
      & ~st_mask
      & ~np.isnan(reversal)
    )
    return np.where(base_valid, reversal, np.nan)
