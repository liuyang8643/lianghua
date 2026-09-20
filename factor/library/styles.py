"""Complementary price styles; eligibility remains owned by env.

Every score at T consumes completed observations only. Short styles require
complete windows; long momentum skips bounded interior gaps in a fixed window.
The implementation deliberately does not consume open, ST, or listing state.
"""

from __future__ import annotations

import numpy as np

from .completed_windows import completed_finite_window_sum


def _matrices(panel: dict, *fields: str) -> tuple[np.ndarray, ...]:
    matrices = tuple(np.asarray(panel[field], dtype=np.float64) for field in fields)
    if any(matrix.ndim != 2 for matrix in matrices):
        raise ValueError("factor inputs must be date-by-stock matrices")
    if any(matrix.shape != matrices[0].shape for matrix in matrices[1:]):
        raise ValueError("factor inputs must have identical shapes")
    return matrices


def _official_log_returns(close: np.ndarray, pre_close: np.ndarray) -> np.ndarray:
    valid = np.isfinite(close) & (close > 0) & np.isfinite(pre_close) & (pre_close > 0)
    result = np.full(close.shape, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(close, pre_close, out=result, where=valid)
        np.log(result, out=result)
    # Only extreme finite prices require the more expensive difference of
    # logs; normal market panels take one in-place logarithm.
    exceptional = valid & ~np.isfinite(result)
    if np.any(exceptional):
        result[exceptional] = np.log(close[exceptional]) - np.log(pre_close[exceptional])
    return result


class CompletedReversal20:
    """Negative compounded official return over [T-20, T)."""

    hist_days = 20
    pre_ranked = False
    requires_full_history = False

    def calc_batch(self, panel: dict) -> np.ndarray:
        close, pre_close = _matrices(panel, "close", "preClose")
        return self.calc_from_returns(_official_log_returns(close, pre_close))

    def calc_from_returns(self, returns: np.ndarray) -> np.ndarray:
        completed = completed_finite_window_sum(returns, self.hist_days)
        with np.errstate(over="ignore", invalid="ignore"):
            return -np.expm1(completed)


class CompletedMomentum252Skip21:
    """Compound observed official returns over the fixed [T-252, T-21).

    Both endpoints and at least 185 of 231 rows (ceil(80%)) must be valid.
    Interior missing returns contribute no log return, without filling raw
    prices, extending the calendar window or extrapolating observed returns.
    This neutral convention does not reconstruct unobserved market returns.
    """

    hist_days = 252
    min_valid_days = 185
    pre_ranked = False
    requires_full_history = False

    def calc_batch(self, panel: dict) -> np.ndarray:
        close, pre_close = _matrices(panel, "close", "preClose")
        return self.calc_from_returns(_official_log_returns(close, pre_close))

    def calc_from_returns(self, returns: np.ndarray) -> np.ndarray:
        completed = completed_finite_window_sum(returns, 231, 21, min_count=self.min_valid_days)
        length = max(0, len(returns) - self.hist_days)
        endpoints = np.isfinite(returns[:length]) & np.isfinite(returns[230 : 230 + length])
        completed[self.hist_days:] = np.where(endpoints, completed[self.hist_days:], np.nan)
        with np.errstate(over="ignore", invalid="ignore"):
            return np.expm1(completed)


class CompletedAmountImbalance20:
    """Completed 20-day signed amount divided by total amount, in [-1, 1].

    Direction is sign(close - official preClose). Flat days retain their
    amount in the denominator. Every amount must be strictly positive.
    """

    hist_days = 20
    pre_ranked = False
    requires_full_history = False

    def calc_batch(self, panel: dict) -> np.ndarray:
        close, pre_close, amount = _matrices(panel, "close", "preClose", "amount")
        valid = (
            np.isfinite(close) & (close > 0)
            & np.isfinite(pre_close) & (pre_close > 0)
            & np.isfinite(amount) & (amount > 0)
        )
        completed_amount = np.where(valid, amount, np.nan)
        direction = (close > pre_close).astype(np.int8) - (close < pre_close)
        signed = completed_finite_window_sum(completed_amount * direction, 20)
        total = completed_finite_window_sum(completed_amount, 20)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = signed / total
        result[~np.isfinite(result) | (total <= 0)] = np.nan
        return np.clip(result, -1.0, 1.0)


class CompletedCloseLocation20:
    """Mean (close-low)/(high-low) over [T-20, T), in [0, 1].

    A zero range or an inconsistent/nonpositive completed bar is missing.
    Averaging completed locations captures persistent closing pressure.
    """

    hist_days = 20
    pre_ranked = False
    requires_full_history = False

    def calc_batch(self, panel: dict) -> np.ndarray:
        high, low, close = _matrices(panel, "high", "low", "close")
        valid = (
            np.isfinite(high) & np.isfinite(low) & np.isfinite(close)
            & (low > 0) & (high > low) & (close >= low) & (close <= high)
        )
        location = np.full(close.shape, np.nan, dtype=np.float64)
        np.divide(close - low, high - low, out=location, where=valid)
        return completed_finite_window_sum(location, 20) / 20.0
