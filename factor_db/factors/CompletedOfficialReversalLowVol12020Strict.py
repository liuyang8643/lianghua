"""Strict completed official-return reversal and low-volatility score."""

from __future__ import annotations

import numpy as np

from factor.library.completed_windows import (
    completed_window_all,
    completed_window_sum,
)


_MOMENTUM_WINDOW = 120
_VOLATILITY_WINDOW = 20


def _official_return_terms(
    close: np.ndarray,
    pre_close: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build log returns, squared returns, and their strict validity mask."""
    valid = (
        np.isfinite(close)
        & (close > 0.0)
        & np.isfinite(pre_close)
        & (pre_close > 0.0)
    )
    daily_return = np.zeros(close.shape, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(
            close,
            pre_close,
            out=daily_return,
            where=valid,
        )
        daily_return -= 1.0

    one_plus_return = 1.0 + daily_return
    valid &= (
        np.isfinite(daily_return)
        & np.isfinite(one_plus_return)
        & (one_plus_return > 0.0)
    )

    log_return = np.zeros(close.shape, dtype=np.float64)
    squared_return = np.zeros(close.shape, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.log1p(
            daily_return,
            out=log_return,
            where=valid,
        )
        np.square(
            daily_return,
            out=squared_return,
            where=valid,
        )

    valid &= np.isfinite(log_return) & np.isfinite(squared_return)
    # Invalid terms are algebraically neutral only inside the accumulators;
    # the exact observation counts below still make every affected score NaN.
    log_return[~valid] = 0.0
    squared_return[~valid] = 0.0
    return log_return, squared_return, valid


class CompletedOfficialReversalLowVol12020Strict:
    """Prefer 120-day reversal and low 20-day official-return volatility.

    Output row ``T`` uses exactly the completed official daily returns
    ``r[d] = close[d] / preClose[d] - 1`` for ``d in [T-120, T)``.
    ``M120`` compounds all 120 returns through
    ``exp(sum(log1p(r))) - 1``.  ``sigma20`` is the root mean square of the
    final 20 returns in that same completed window.  The score is

    ``0.25 * clip((0.40 - M120) / 0.80, 0, 1)``
    ``+ 0.75 * clip((0.06 - sigma20) / 0.05, 0, 1)``.

    Every close, official pre-close, return, and derived term must be finite;
    prices and ``1 + r`` must be strictly positive.  One invalid observation
    invalidates every exact 120-row window containing it.  No current-row
    value or market field other than close and preClose enters the score.
    """

    hist_days = _MOMENTUM_WINDOW
    update_frequency = "daily"
    pre_ranked = False
    requires_full_history = False

    def calc_batch(self, panel: dict) -> np.ndarray:
        close = np.asarray(panel["close"], dtype=np.float64)
        pre_close = np.asarray(panel["preClose"], dtype=np.float64)
        if close.ndim != 2 or pre_close.ndim != 2:
            raise ValueError("close and preClose must be two-dimensional")
        if close.shape != pre_close.shape:
            raise ValueError("close and preClose must have matching shapes")

        rows, stocks = close.shape
        result = np.full((rows, stocks), np.nan, dtype=np.float32)
        if rows <= _MOMENTUM_WINDOW:
            return result

        log_return, squared_return, daily_valid = _official_return_terms(
            close,
            pre_close,
        )
        total_log_return = completed_window_sum(
            log_return,
            _MOMENTUM_WINDOW,
        )
        total_squared_return = completed_window_sum(
            squared_return,
            _VOLATILITY_WINDOW,
        )[_MOMENTUM_WINDOW - _VOLATILITY_WINDOW :]
        long_valid = completed_window_all(
            daily_valid.copy(),
            _MOMENTUM_WINDOW,
        )
        volatility_valid = completed_window_all(
            daily_valid,
            _VOLATILITY_WINDOW,
        )[_MOMENTUM_WINDOW - _VOLATILITY_WINDOW :]

        with np.errstate(
            divide="ignore",
            invalid="ignore",
            over="ignore",
            under="ignore",
        ):
            np.exp(total_log_return, out=total_log_return)
            total_log_return -= 1.0
            total_squared_return /= _VOLATILITY_WINDOW
            np.sqrt(total_squared_return, out=total_squared_return)
            reversal120 = np.clip(
                (0.40 - total_log_return) / 0.80,
                0.0,
                1.0,
            )
            low_vol20 = np.clip(
                (0.06 - total_squared_return) / 0.05,
                0.0,
                1.0,
            )
            score = 0.25 * reversal120 + 0.75 * low_vol20

        valid_score = (
            long_valid
            & volatility_valid
            & np.isfinite(total_log_return)
            & np.isfinite(total_squared_return)
            & np.isfinite(score)
        )
        result[_MOMENTUM_WINDOW:] = np.where(
            valid_score,
            score,
            np.nan,
        ).astype(np.float32)

        return result
