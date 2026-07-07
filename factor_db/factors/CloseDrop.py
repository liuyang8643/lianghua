import numpy as np

MIN_PRICE = 2.0


class CloseDrop:
    """收盘跌幅因子：当日跌幅越大分数越高（逆向/接飞刀）。

    score = (preClose - close) / preClose = 1 - close/preClose
    收盘相对前收越低，分数越高，排名越靠前。
    使用 close[T] + preClose[T]，盘后信号合法。
    """
    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        close = panel["close"]
        pre_close = panel["preClose"]
        st_mask = panel["st_mask"]

        valid = (
            ~np.isnan(close)
            & (close >= MIN_PRICE)
            & ~np.isnan(pre_close)
            & (pre_close > 0)
            & ~st_mask
        )

        drop = (pre_close - close) / pre_close  # >0=跌, <0=涨
        return np.where(valid, drop.astype(np.float32), np.nan)
