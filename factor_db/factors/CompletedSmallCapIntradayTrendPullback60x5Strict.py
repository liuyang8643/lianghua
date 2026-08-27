"""Strict completed-day small-cap intraday trend/pullback interaction."""

from __future__ import annotations

import numpy as np


_LONG_WINDOW = 60
_SHORT_WINDOW = 5
_SIZE_PIVOT_RMB = 1.0e10
_SCORE_CAP_TO_PIVOT_RATIO = 4.0e4 / _SIZE_PIVOT_RMB


def _intraday_terms(
    open_price: np.ndarray,
    close: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``log(close/open)`` and its strict observation mask."""
    valid = np.greater(open_price, 0.0)
    exceptional = np.empty(open_price.shape, dtype=bool)
    np.greater(close, 0.0, out=exceptional)
    np.logical_and(valid, exceptional, out=valid)
    values = np.zeros(open_price.shape, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(
            close,
            open_price,
            out=values,
            where=valid,
        )
        np.log(values, out=values, where=valid)

    np.isfinite(values, out=exceptional)
    np.logical_not(exceptional, out=exceptional)
    np.logical_and(valid, exceptional, out=exceptional)
    if np.any(exceptional):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            replacement = (
                np.log(close[exceptional])
                - np.log(open_price[exceptional])
            )
        recovered = np.isfinite(replacement)
        values[exceptional] = np.where(recovered, replacement, 0.0)
        valid[exceptional] = recovered
    return values, valid


def _calculate_rolling(
    open_price: np.ndarray,
    close: np.ndarray,
    total_share: np.ndarray,
    result: np.ndarray,
) -> None:
    rows, width = close.shape
    daily, daily_valid = _intraday_terms(open_price, close)
    long_count = np.sum(
        daily_valid[:_LONG_WINDOW],
        axis=0,
        dtype=np.int16,
    )
    rolling_sum = np.empty((2, width), dtype=np.float64)
    rolling_energy = np.empty((2, width), dtype=np.float64)
    rolling_sum[0] = np.sum(
        daily[:_LONG_WINDOW],
        axis=0,
        dtype=np.float64,
    )
    rolling_energy[0] = np.sum(
        daily[:_LONG_WINDOW] * daily[:_LONG_WINDOW],
        axis=0,
        dtype=np.float64,
    )
    rolling_sum[1] = np.sum(
        daily[_LONG_WINDOW - _SHORT_WINDOW : _LONG_WINDOW],
        axis=0,
        dtype=np.float64,
    )
    rolling_energy[1] = np.sum(
        daily[_LONG_WINDOW - _SHORT_WINDOW : _LONG_WINDOW]
        * daily[_LONG_WINDOW - _SHORT_WINDOW : _LONG_WINDOW],
        axis=0,
        dtype=np.float64,
    )
    window_lengths = np.asarray(
        [[_LONG_WINDOW], [_SHORT_WINDOW]],
        dtype=np.float64,
    )
    denominator = np.empty((2, width), dtype=np.float64)
    score = np.empty((2, width), dtype=np.float64)
    long_score = score[0]
    short_score = score[1]
    smallness = np.empty(width, dtype=np.float64)
    positive_energy = np.empty((2, width), dtype=bool)
    size_valid = np.empty(width, dtype=bool)
    valid = np.empty(width, dtype=bool)

    for row in range(_LONG_WINDOW, rows):
        np.multiply(
            rolling_energy,
            window_lengths,
            out=denominator,
        )
        np.greater(rolling_energy, 0.0, out=positive_energy)
        np.sqrt(
            denominator,
            out=denominator,
            where=positive_energy,
        )

        score.fill(0.0)
        np.divide(
            rolling_sum,
            denominator,
            out=score,
            where=positive_energy,
        )
        long_score += 1.0
        np.subtract(1.0, short_score, out=short_score)
        long_score *= short_score

        last_close = close[row - 1]
        last_total_share = total_share[row - 1]
        np.isfinite(last_total_share, out=size_valid)
        np.greater(last_total_share, 0.0, out=valid)
        size_valid &= valid
        np.multiply(last_close, last_total_share, out=smallness)
        smallness *= _SCORE_CAP_TO_PIVOT_RATIO
        smallness += 4.0
        np.divide(long_score, smallness, out=long_score)

        np.equal(long_count, _LONG_WINDOW, out=valid)
        valid &= size_valid
        np.copyto(
            result[row],
            long_score,
            where=valid,
            casting="unsafe",
        )

        if row + 1 < rows:
            long_outgoing = row - _LONG_WINDOW
            short_outgoing = row - _SHORT_WINDOW
            long_count += daily_valid[row]
            long_count -= daily_valid[long_outgoing]
            rolling_sum += daily[row]
            rolling_sum[0] -= daily[long_outgoing]
            rolling_sum[1] -= daily[short_outgoing]
            scratch = denominator[0]
            np.square(daily[row], out=scratch)
            rolling_energy += scratch
            np.square(
                daily[long_outgoing],
                out=scratch,
            )
            rolling_energy[0] -= scratch
            np.square(
                daily[short_outgoing],
                out=scratch,
            )
            rolling_energy[1] -= scratch


class CompletedSmallCapIntradayTrendPullback60x5Strict:
    """Prefer small stocks in long intraday uptrends with a short pullback.

    Output row ``T`` uses completed rows only.  For every ``d < T``,
    ``r[d] = log(close[d] / open[d])``; the return deliberately excludes the
    ``open/preClose`` overnight gap.  On exact windows ``[T-60, T)`` and
    ``[T-5, T)``, respectively,

    ``Q60 = sum(r) / sqrt(60 * sum(r**2))``
    ``Q5 = sum(r) / sqrt(5 * sum(r**2))``.

    A complete zero-energy window has ``Q = 0``.  Smallness is based only on
    the last completed market value,
    ``S = 1 / (1 + close[T-1] * total_share[T-1] * 10000 / 1e10)``,
    where runtime ``total_share`` is measured in ten-thousand shares.  The
    100亿元 RMB pivot is fixed and is not a searched parameter.  The final
    score is
    ``S * ((1 + Q60) / 2) * ((1 - Q5) / 2)``.  Since each consistency
    ratio is in ``[-1, 1]``, this fixed transformation continuously rewards
    higher long-window consistency and lower short-window consistency without
    introducing a zero-score plateau.

    All 60 open/close pairs and the last completed close/total_share pair
    must be finite and strictly positive.  Missing observations are exposed
    as NaN and are never filled or shortened.  No current-row field,
    preClose, high, low, volume, amount, ST state, gap, or external data is
    read.
    """

    hist_days = _LONG_WINDOW
    update_frequency = "daily"
    pre_ranked = False
    requires_full_history = False

    def calc_batch(self, panel: dict) -> np.ndarray:
        open_price = np.asarray(panel["open"])
        close = np.asarray(panel["close"])
        total_share = np.asarray(panel["total_share"])
        if not (
            open_price.ndim == close.ndim == total_share.ndim == 2
        ):
            raise ValueError(
                "open, close, and total_share must be two-dimensional"
            )
        if not (open_price.shape == close.shape == total_share.shape):
            raise ValueError(
                "open, close, and total_share must have matching shapes"
            )
        if any(
            values.dtype.kind not in "iuf"
            for values in (open_price, close, total_share)
        ):
            raise ValueError(
                "open, close, and total_share must have real numeric dtypes"
            )
        open_price = open_price.astype(np.float64, copy=False)
        close = close.astype(np.float64, copy=False)
        total_share = total_share.astype(np.float64, copy=False)

        rows, stocks = close.shape
        result = np.full((rows, stocks), np.nan, dtype=np.float32)
        if rows <= _LONG_WINDOW:
            return result

        _calculate_rolling(
            open_price,
            close,
            total_share,
            result,
        )

        return result
