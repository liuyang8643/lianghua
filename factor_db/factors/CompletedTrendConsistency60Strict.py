"""Positive trend consistency from strictly completed official returns."""

from __future__ import annotations

import numpy as np


_WINDOW = 60
_REBASE_INTERVAL = 4096


def _official_log_returns(
    close: np.ndarray,
    pre_close: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return robust official log returns and their strict validity mask."""
    log_return = np.empty(close.shape, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(close, pre_close, out=log_return)
        np.log(log_return, out=log_return)

    # Division can overflow or underflow for otherwise valid float64 price
    # pairs.  The log-difference identity preserves the requested mathematical
    # return for those exceptional observations without paying for a second
    # full-panel logarithm in the normal path.
    # A finite logarithm plus two strictly-positive inputs is equivalent to
    # the full finite/positive pair contract in the normal path: any infinity
    # in either input necessarily produces a non-finite ratio logarithm.
    positive = (close > 0.0) & (pre_close > 0.0)
    finite_log = np.isfinite(log_return)
    exceptional = positive & ~finite_log
    if np.any(exceptional):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            log_return[exceptional] = (
                np.log(close[exceptional])
                - np.log(pre_close[exceptional])
            )
        finite_log[exceptional] = np.isfinite(log_return[exceptional])

    valid = positive
    np.logical_and(valid, finite_log, out=valid)
    log_return[~valid] = 0.0
    return log_return, valid


class CompletedTrendConsistency60Strict:
    """Emit only consistent positive trends from 60 completed trading days.

    Output row ``T`` uses exactly the completed rows ``d in [T-60, T)``.
    Each official log return is
    ``ell[d] = log(close[d] / preClose[d])``, evaluated with a numerically
    robust log difference only when the direct ratio leaves the float64
    range.  Every one of the 60 close/preClose pairs must be finite and
    strictly positive.  The fixed score is

    ``Q = sum(ell) / sqrt(60 * sum(ell**2))``.

    A score is emitted only when ``Q > 0``; non-positive or zero-energy
    trends remain NaN so the trend sleeve can hold cash.  No open price,
    current-row HLCVA, ST state, gap, or external data is read.
    """

    hist_days = _WINDOW
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
        if rows <= _WINDOW:
            return result

        log_return, daily_valid = _official_log_returns(close, pre_close)
        rolling_count = np.sum(
            daily_valid[:_WINDOW],
            axis=0,
            dtype=np.int16,
        )
        rolling_sum = np.sum(
            log_return[:_WINDOW],
            axis=0,
            dtype=np.float64,
        )
        rolling_energy = np.sum(
            log_return[:_WINDOW] * log_return[:_WINDOW],
            axis=0,
            dtype=np.float64,
        )
        denominator = np.empty(stocks, dtype=np.float64)
        score = np.empty(stocks, dtype=np.float64)

        for row in range(_WINDOW, rows):
            if row > _WINDOW and row % _REBASE_INTERVAL == 0:
                start = row - _WINDOW
                history = log_return[start:row]
                rolling_count = np.sum(
                    daily_valid[start:row],
                    axis=0,
                    dtype=np.int16,
                )
                rolling_sum = np.sum(
                    history,
                    axis=0,
                    dtype=np.float64,
                )
                rolling_energy = np.sum(
                    history * history,
                    axis=0,
                    dtype=np.float64,
                )

            np.multiply(rolling_energy, _WINDOW, out=denominator)
            with np.errstate(invalid="ignore", over="ignore"):
                np.sqrt(denominator, out=denominator)
            valid_score = (
                (rolling_count == _WINDOW)
                & (rolling_energy > 0.0)
                & (rolling_sum > 0.0)
            )
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                np.divide(
                    rolling_sum,
                    denominator,
                    out=score,
                )
            np.copyto(
                result[row],
                score,
                where=valid_score,
                casting="unsafe",
            )

            if row + 1 < rows:
                outgoing = row - _WINDOW
                rolling_count += daily_valid[row]
                rolling_count -= daily_valid[outgoing]
                rolling_sum += log_return[row]
                rolling_sum -= log_return[outgoing]
                with np.errstate(invalid="ignore", over="ignore"):
                    rolling_energy += log_return[row] * log_return[row]
                    rolling_energy -= (
                        log_return[outgoing] * log_return[outgoing]
                    )

        return result
