"""TRIX三重平滑趋势因子 — 原始值."""
import numpy as np
from scipy.signal import lfilter

MIN_PRICE = 2.0


def _ema_2d(data, period):
    alpha = 2.0 / (period + 1.0)
    b = [alpha]
    a = [1.0, -(1.0 - alpha)]
    filled = np.nan_to_num(data, nan=0.0)
    return lfilter(b, a, filled, axis=0)


class TRIX:
    hist_days = 50

    def calc_batch(self, panel: dict) -> np.ndarray:
        close = panel["close"]
        opn = panel["open"]
        st = panel["st_mask"]
        valid = ~np.isnan(opn) & (opn >= MIN_PRICE) & ~st

        cp = np.roll(close, 1, axis=0); cp[0] = np.nan
        ema1 = _ema_2d(cp, 12)
        ema2 = _ema_2d(ema1, 12)
        ema3 = _ema_2d(ema2, 12)

        trix = np.full_like(close, np.nan)
        trix[1:] = (ema3[1:] - ema3[:-1]) / np.where(np.abs(ema3[:-1]) > 1e-9, ema3[:-1], np.nan) * 100.0

        return np.where(valid, trix, np.nan)
