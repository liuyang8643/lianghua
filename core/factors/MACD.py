"""MACD趋势因子 — 直方图原始值."""
import numpy as np
from scipy.signal import lfilter

MIN_PRICE = 2.0


def _ema_2d(data, period):
    alpha = 2.0 / (period + 1.0)
    b = [alpha]
    a = [1.0, -(1.0 - alpha)]
    filled = np.nan_to_num(data, nan=0.0)
    return lfilter(b, a, filled, axis=0)


class MACD:
    hist_days = 34

    def calc_batch(self, panel: dict) -> np.ndarray:
        close = panel["close"]
        opn = panel["open"]
        st = panel["st_mask"]
        valid = ~np.isnan(opn) & (opn >= MIN_PRICE) & ~st

        cp = np.roll(close, 1, axis=0); cp[0] = np.nan
        ema12 = _ema_2d(cp, 12)
        ema26 = _ema_2d(cp, 26)
        dif = ema12 - ema26
        dea = _ema_2d(dif, 9)
        hist = 2.0 * (dif - dea)

        return np.where(valid, hist, np.nan)
