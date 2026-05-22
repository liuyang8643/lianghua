import numpy as np

class QualityReversal10D:
  """质量反转: 10日超跌+高利润增长的小盘股."""

  hist_days = 10

  def calc_batch(self, panel: dict) -> np.ndarray:
    raw_open = panel["open"]
    total_share = panel["total_share"]
    st_mask = panel["st_mask"]
    profit_yoy = panel["profit_yoy"]

    base_valid = (
      ~np.isnan(raw_open) & (raw_open >= 2.0)
      & ~np.isnan(total_share) & (total_share > 0)
      & ~st_mask
    )

    total_mv_yi = (raw_open * total_share) / 1e8

    n_dates, n_stocks = len(panel["trade_dates"]), len(panel["stock_codes"])
    ret_10d = np.full((n_dates, n_stocks), np.nan, dtype=np.float32)
    ret_10d[10:] = raw_open[9:-1] / raw_open[:-10] - 1.0
    rev_raw = -ret_10d

    size_val = -total_mv_yi
    rev_contrib = np.where(base_valid & ~np.isnan(rev_raw), 1 + 0.40 * rev_raw, 1.0)
    py_contrib = np.where(base_valid & ~np.isnan(profit_yoy), 1 + 0.20 * profit_yoy, 1.0)

    score = size_val * rev_contrib * py_contrib
    return np.where(base_valid, score, np.nan)
