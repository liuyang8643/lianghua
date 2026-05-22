import numpy as np

MIN_RAW_PRICE = 2.0


class ShareExpansion:
  """股本扩张 — total_share年增速，大幅增发=投资激进=低收益"""
  hist_days = 252

  def calc_batch(self, panel: dict) -> np.ndarray:
    total_share = panel["total_share"]
    raw_open = panel["open"]
    st_mask = panel["st_mask"]

    w = self.hist_days
    expansion = np.empty_like(total_share, dtype=float)
    expansion[:w] = np.nan
    expansion[w:] = total_share[w:] / total_share[:-w] - 1.0

    base_valid = (
      ~np.isnan(raw_open)
      & (raw_open >= MIN_RAW_PRICE)
      & ~st_mask
      & ~np.isnan(expansion)
    )
    return np.where(base_valid, -expansion, np.nan)
