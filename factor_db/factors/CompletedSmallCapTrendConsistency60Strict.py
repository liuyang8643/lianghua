"""Strict completed-day trend-consistency and small-cap interaction."""

from __future__ import annotations

import numpy as np


_WINDOW = 60
_REBASE_INTERVAL = 4096
_SIZE_PIVOT_RMB = 1.0e10
_CAP_TO_PIVOT_RATIO = 1.0e4 / _SIZE_PIVOT_RMB


def _daily_log_return(
    close: np.ndarray,
    pre_close: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return official log returns and their strict validity mask."""
    values = np.empty(close.shape, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(close, pre_close, out=values)
        np.log(values, out=values)

    positive = (close > 0.0) & (pre_close > 0.0)
    finite_log = np.isfinite(values)
    exceptional = np.empty(close.shape, dtype=bool)
    np.logical_not(finite_log, out=exceptional)
    np.logical_and(positive, exceptional, out=exceptional)
    if np.any(exceptional):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            values[exceptional] = (
                np.log(close[exceptional])
                - np.log(pre_close[exceptional])
            )
        finite_log[exceptional] = np.isfinite(values[exceptional])

    valid = positive
    np.logical_and(valid, finite_log, out=valid)
    values[~valid] = 0.0
    return values, valid


class CompletedSmallCapTrendConsistency60Strict:
    """Prefer persistent positive trends specifically among smaller stocks.

    Row ``T`` uses exactly the 60 completed rows ``[T-60, T)``.  For each
    completed row ``d``, the official log return is
    ``ell[d] = log(close[d]) - log(preClose[d])``.  All 60 close/preClose
    pairs must be finite and strictly positive.

    Trend consistency is the signed mean-to-RMS ratio
    ``Q = sum(ell) / sqrt(60 * sum(ell**2))``.  An exactly flat window has
    ``Q = 0``.  Smallness uses only the last completed market value,
    ``cap = close[T-1] * total_share[T-1] * 10000``, because runtime
    ``total_share`` is measured in ten-thousand shares, through
    ``S = 1 / (1 + cap / 1e10)``.  The 100亿元 RMB pivot is a fixed unit
    scale and is not a searched parameter.  The final score is ``Q * S``:
    persistent uptrends in small stocks score highly, while persistent
    downtrends in small stocks receive a stronger penalty.

    No current-row field, open price, gap, high, low, volume, amount, ST
    state, or external data is read.  Missing inputs invalidate the affected
    window; no observation is filled, skipped, or used in a shorter window.
    """

    hist_days = _WINDOW
    update_frequency = "daily"
    pre_ranked = False
    requires_full_history = False

    def calc_batch(self, panel: dict) -> np.ndarray:
        close = np.asarray(panel["close"])
        pre_close = np.asarray(panel["preClose"])
        total_share = np.asarray(panel["total_share"])
        if close.ndim != 2:
            raise ValueError(
                "close, preClose, and total_share must be two-dimensional"
            )
        if not (close.shape == pre_close.shape == total_share.shape):
            raise ValueError(
                "close, preClose, and total_share must have matching shapes"
            )
        if any(
            values.dtype.kind not in "iuf"
            for values in (close, pre_close, total_share)
        ):
            raise ValueError(
                "close, preClose, and total_share must have real numeric dtypes"
            )

        rows, stocks = close.shape
        result = np.full((rows, stocks), np.nan, dtype=np.float32)
        if rows <= _WINDOW:
            return result

        daily, daily_valid = _daily_log_return(close, pre_close)
        rolling_count = np.sum(
            daily_valid[:_WINDOW],
            axis=0,
            dtype=np.int16,
        )
        rolling_sum = np.sum(
            daily[:_WINDOW],
            axis=0,
            dtype=np.float64,
        )
        rolling_energy = np.sum(
            daily[:_WINDOW] * daily[:_WINDOW],
            axis=0,
            dtype=np.float64,
        )
        denominator = np.empty(stocks, dtype=np.float64)
        score = np.empty(stocks, dtype=np.float64)
        smallness = np.empty(stocks, dtype=np.float64)

        for row in range(_WINDOW, rows):
            if row > _WINDOW and row % _REBASE_INTERVAL == 0:
                start = row - _WINDOW
                history = daily[start:row]
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
            score.fill(0.0)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                np.divide(
                    rolling_sum,
                    denominator,
                    out=score,
                    where=rolling_energy > 0.0,
                )

            last_close = close[row - 1]
            last_total_share = total_share[row - 1]
            size_valid = (
                np.isfinite(last_close)
                & (last_close > 0.0)
                & np.isfinite(last_total_share)
                & (last_total_share > 0.0)
            )
            with np.errstate(invalid="ignore", over="ignore"):
                np.multiply(last_close, last_total_share, out=smallness)
                smallness *= _CAP_TO_PIVOT_RATIO
                smallness += 1.0
                np.reciprocal(smallness, out=smallness)
                score *= smallness

            valid = (
                (rolling_count == _WINDOW)
                & size_valid
                & np.isfinite(rolling_sum)
                & np.isfinite(denominator)
                & np.isfinite(score)
            )
            np.copyto(
                result[row],
                score,
                where=valid,
                casting="unsafe",
            )

            if row + 1 < rows:
                outgoing = row - _WINDOW
                rolling_count += daily_valid[row]
                rolling_count -= daily_valid[outgoing]
                rolling_sum += daily[row]
                rolling_sum -= daily[outgoing]
                with np.errstate(invalid="ignore", over="ignore"):
                    rolling_energy += daily[row] * daily[row]
                    rolling_energy -= daily[outgoing] * daily[outgoing]

        return result
