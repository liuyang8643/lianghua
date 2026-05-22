"""历史相对跳空因子 — 当前跳空 vs 20日均值的偏离."""
import numpy as np
from core.factors.helpers import BaseFactor
from core.factors.helpers.rolling import roll_mean

MIN_PRICE = 2.0


class HistoricalGap(BaseFactor):
    """当前跳空 vs 20日历史均值: 相对跳空越低分越高."""
    hist_days = 23

    def calc_batch(self, panel: dict) -> np.ndarray:
        opn = panel["open"]
        close = panel["close"]
        st = panel["st_mask"]
        valid = ~np.isnan(opn) & (opn >= MIN_PRICE) & ~st
        N, S = opn.shape

        # 每日跳空
        daily_gap = np.full((N, S), np.nan)
        daily_gap[1:] = opn[1:] / close[:-1] - 1.0

        # 20日平均跳空
        gap_ma20 = roll_mean(daily_gap, 20)

        # 相对跳空 = 当日跳空 - 20日均值
        relative_gap = daily_gap - gap_ma20

        # score = max(0, -relative_gap * 10) — 向下偏离均值越多分越高
        score = np.maximum(0.0, -relative_gap * 10.0)

        return np.where(valid, score, np.nan)
