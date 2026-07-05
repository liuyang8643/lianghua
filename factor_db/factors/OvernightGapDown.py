import numpy as np

MIN_RAW_PRICE = 2.0


class OvernightGapDown:
    """隔夜低开因子：低开越多分数越高。

    score = -(open/preClose - 1) = 1 - open/preClose
    开盘相对前收越低（跳空低开越多），分数越高。
    仅使用 open[T] + preClose[T]，不涉及 T 日盘中价格，符合红线。
    """
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        raw_open = panel["open"]
        pre_close = panel["preClose"]
        st_mask = panel["st_mask"]

        valid = (
            ~np.isnan(raw_open)
            & (raw_open >= MIN_RAW_PRICE)
            & ~np.isnan(pre_close)
            & (pre_close > 0)
            & ~st_mask
        )

        gap = raw_open / pre_close - 1.0
        score = -gap  # 低开越多 → gap 越负 → score 越高

        return np.where(valid, score.astype(np.float32), np.nan)
