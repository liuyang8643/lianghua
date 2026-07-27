"""Intermediate 12-minus-1 momentum from completed adjusted returns."""

import numpy as np


def _lag(values: np.ndarray, periods: int) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=values.dtype)
    if periods < len(values):
        result[periods:] = values[:-periods]
    return result


def _rolling_sum_complete(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling sum that is finite only when the whole window is finite."""
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    cumulative = np.cumsum(
        np.where(finite, values, 0.0), axis=0, dtype=np.float32,
    )
    counts = np.cumsum(finite, axis=0, dtype=np.int16)
    cumulative[window:] -= cumulative[:-window]
    counts[window:] -= counts[:-window]
    cumulative[:window - 1] = np.nan
    cumulative[window - 1:][counts[window - 1:] != window] = np.nan
    return cumulative


class IntermediateMomentum12Minus1:
    """Favor 12-month winners while excluding the most recent 21 days."""

    hist_days = 252
    pre_ranked = False
    requires_full_history = False

    def calc_batch(self, panel: dict) -> np.ndarray:
        close = np.asarray(panel["close"], dtype=np.float32)
        pre_close = np.asarray(panel["preClose"], dtype=np.float32)
        open_price = np.asarray(panel["open"], dtype=np.float32)
        st_mask = np.asarray(panel["st_mask"], dtype=bool)

        with np.errstate(divide="ignore", invalid="ignore"):
            daily_ratio = close / pre_close
            daily_log_return = np.where(
                np.isfinite(daily_ratio) & (daily_ratio > 0.0),
                np.log(daily_ratio),
                np.nan,
            )
            completed_12_minus_1 = _lag(
                _rolling_sum_complete(daily_log_return, 231), 21,
            )
            score = np.expm1(completed_12_minus_1)

        valid = (
            np.isfinite(score)
            & np.isfinite(open_price)
            & (open_price >= 2.0)
            & ~st_mask
        )
        return np.where(valid, score, np.nan).astype(np.float32, copy=False)
