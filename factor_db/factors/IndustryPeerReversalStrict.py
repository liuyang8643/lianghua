"""Strict point-in-time industry peer mean-reversion factors.

The concrete classes in this module differ only in their fixed estimation
and recent-deviation windows.  They deliberately have no constructor
parameters so the normal factor registry and GA can discover and instantiate
them without adding a parameter channel to the backtest framework.
"""

from __future__ import annotations

import numpy as np


_MIN_PEERS = 3
_MIN_CORRELATION = 0.20


def _official_log_returns(
    close: np.ndarray,
    pre_close: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return completed official log returns and a strict validity mask."""
    log_return = np.empty(close.shape, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(close, pre_close, out=log_return)
        np.log(log_return, out=log_return)

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

    valid = positive & finite_log
    log_return[~valid] = 0.0
    return log_return, valid


def _leave_one_out_peer_returns(
    log_return: np.ndarray,
    return_valid: np.ndarray,
    industry_id: np.ndarray,
    min_peers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute daily equal-weight peer returns without a per-stock loop.

    Industry ids are compressed once for the whole panel.  A flattened
    ``row * group_count + group`` key then lets two ``np.bincount`` calls
    aggregate every date and industry together.  A candidate's own return is
    subtracted from its group total and count.  The implementation has no
    Python date or stock loop and retains exact point-in-time membership.
    """
    rows, stocks = log_return.shape
    known_industry = np.isfinite(industry_id) & (industry_id >= 0)

    encoded_values = np.unique(industry_id[known_industry])
    if encoded_values.size == 0:
        return (
            np.zeros((rows, stocks), dtype=np.float64),
            np.zeros((rows, stocks), dtype=bool),
        )
    group_total = int(encoded_values.size)
    index_dtype = np.int16 if group_total < np.iinfo(np.int16).max else np.int32

    integer_fast_path = (
        np.issubdtype(industry_id.dtype, np.integer)
        and int(np.min(industry_id)) >= -1
        and int(encoded_values[-1]) <= 10_000_000
    )
    if integer_fast_path:
        lookup = np.full(
            int(encoded_values[-1]) + 2,
            group_total,
            dtype=index_dtype,
        )
        lookup[encoded_values.astype(np.intp, copy=False)] = np.arange(
            group_total,
            dtype=index_dtype,
        )
        # Unknown -1 deliberately indexes the final sentinel entry.
        group_index = lookup[industry_id]
    else:
        group_index = np.full(
            industry_id.shape,
            group_total,
            dtype=index_dtype,
        )
        group_index[known_industry] = np.searchsorted(
            encoded_values,
            industry_id[known_industry],
        ).astype(index_dtype, copy=False)

    bins = group_total + 1
    group_key = group_index.astype(np.int32, copy=False)
    if group_key is group_index:
        group_key = group_key.copy()
    del group_index
    group_key += (
        np.arange(rows, dtype=np.int32) * bins
    )[:, np.newaxis]
    source_valid = return_valid & known_industry
    del known_industry
    weighted_return = np.where(source_valid, log_return, 0.0)
    flat_size = rows * bins
    count_dtype = (
        np.int16 if stocks <= np.iinfo(np.int16).max else np.int32
    )
    group_count = np.bincount(
        group_key.ravel(),
        weights=source_valid.ravel(),
        minlength=flat_size,
    ).astype(count_dtype)
    group_sum = np.bincount(
        group_key.ravel(),
        weights=weighted_return.ravel(),
        minlength=flat_size,
    )
    del weighted_return

    peer_count = np.empty((rows, stocks), dtype=count_dtype)
    np.take(group_count, group_key, out=peer_count)
    del group_count
    peer_count -= source_valid
    peer_return = np.empty((rows, stocks), dtype=np.float64)
    np.take(group_sum, group_key, out=peer_return)
    del group_sum, group_key
    peer_return -= log_return
    pair_valid = source_valid & (peer_count >= min_peers)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(
            peer_return,
            peer_count,
            out=peer_return,
            where=pair_valid,
        )
    peer_return[~pair_valid] = 0.0

    return peer_return, pair_valid


class _IndustryPeerReversalStrictBase:
    """Implementation shared by the fixed, registry-discoverable variants."""

    update_frequency = "daily"
    pre_ranked = False
    requires_full_history = False
    min_peers = _MIN_PEERS
    min_correlation = _MIN_CORRELATION

    def calc_batch(self, panel: dict) -> np.ndarray:
        """Return positive scores for recent industry-relative laggards.

        For output row ``T``, let ``N`` be ``estimation_window`` and ``K`` be
        ``deviation_window``.  The factor performs these strictly completed
        operations:

        * fit ``r_i = alpha + beta * r_peer + epsilon`` on
          ``[T-K-N, T-K)``;
        * require positive correlation of at least ``min_correlation``;
        * sum the frozen-model residuals over the independent interval
          ``[T-K, T)``;
        * divide by estimation residual volatility times ``sqrt(K)`` and
          negate, so a relatively underperforming stock has a high score.

        ``r_peer`` is the equal-weight log return of valid stocks with the
        same point-in-time ``industry_id``, excluding the candidate itself.
        Each day needs at least ``min_peers`` such peers.  The candidate's
        industry id must be known (finite and non-negative) and unchanged
        throughout the full ``N + K`` completed rows.  All own and peer
        returns in both windows are required; nothing is filled or clipped.

        No row-T value is read into row-T output.  In particular, row-T
        ``close``, ``preClose``, and ``industry_id`` affect only later rows.
        """
        close = np.asarray(panel["close"])
        pre_close = np.asarray(panel["preClose"])
        industry_id = np.asarray(panel["industry_id"])
        if close.ndim != 2 or pre_close.ndim != 2 or industry_id.ndim != 2:
            raise ValueError(
                "close, preClose, and industry_id must be two-dimensional"
            )
        if close.shape != pre_close.shape or close.shape != industry_id.shape:
            raise ValueError(
                "close, preClose, and industry_id must have matching shapes"
            )
        if not np.issubdtype(close.dtype, np.number) or not np.issubdtype(
            pre_close.dtype,
            np.number,
        ):
            raise ValueError("close and preClose must be numeric panels")
        if not np.issubdtype(industry_id.dtype, np.number):
            raise ValueError("industry_id must be a numeric encoded panel")

        rows, stocks = close.shape
        result = np.full((rows, stocks), np.nan, dtype=np.float32)
        estimation = self.estimation_window
        recent = self.deviation_window
        total_window = estimation + recent
        if rows <= total_window:
            return result

        known_industry = np.isfinite(industry_id) & (industry_id >= 0)
        active_columns = np.flatnonzero(np.any(known_industry, axis=0))
        if active_columns.size == 0:
            return result
        identity_order = (
            active_columns.size == stocks
            and np.array_equal(
                active_columns,
                np.arange(stocks, dtype=active_columns.dtype),
            )
        )
        if identity_order:
            active_close = close
            active_pre_close = pre_close
            active_industry = industry_id
        else:
            # ``np.take`` directly materializes C order.  Advanced column
            # indexing would first create a Fortran-order temporary and then
            # require a second full-panel copy for the date-major hot paths.
            active_close = np.take(close, active_columns, axis=1)
            active_pre_close = np.take(pre_close, active_columns, axis=1)
            active_industry = np.take(industry_id, active_columns, axis=1)
        del known_industry

        log_return, return_valid = _official_log_returns(
            active_close,
            active_pre_close,
        )
        del active_close, active_pre_close
        peer_return, pair_valid = _leave_one_out_peer_returns(
            log_return,
            return_valid,
            active_industry,
            self.min_peers,
        )
        del return_valid

        estimation_sum_x = np.sum(
            peer_return[:estimation],
            axis=0,
            dtype=np.float64,
        )
        estimation_sum_y = np.sum(
            log_return[:estimation],
            axis=0,
            dtype=np.float64,
        )
        estimation_sum_xx = np.sum(
            peer_return[:estimation] * peer_return[:estimation],
            axis=0,
            dtype=np.float64,
        )
        estimation_sum_yy = np.sum(
            log_return[:estimation] * log_return[:estimation],
            axis=0,
            dtype=np.float64,
        )
        estimation_sum_xy = np.sum(
            peer_return[:estimation] * log_return[:estimation],
            axis=0,
            dtype=np.float64,
        )

        recent_sum_x = np.sum(
            peer_return[estimation:total_window],
            axis=0,
            dtype=np.float64,
        )
        recent_sum_y = np.sum(
            log_return[estimation:total_window],
            axis=0,
            dtype=np.float64,
        )

        full_window_valid = np.sum(
            pair_valid[:total_window],
            axis=0,
            dtype=np.int16,
        )
        industry_transition = active_industry[1:] != active_industry[:-1]
        transition_count = np.sum(
            industry_transition[: total_window - 1],
            axis=0,
            dtype=np.int16,
        )

        for output_row in range(total_window, rows):
            eligible = np.flatnonzero(
                (full_window_valid == total_window)
                & (transition_count == 0)
            )
            if eligible.size:
                sum_x = estimation_sum_x[eligible]
                sum_y = estimation_sum_y[eligible]
                centered_xx = (
                    estimation_sum_xx[eligible]
                    - sum_x * sum_x / estimation
                )
                centered_yy = (
                    estimation_sum_yy[eligible]
                    - sum_y * sum_y / estimation
                )
                centered_xy = (
                    estimation_sum_xy[eligible]
                    - sum_x * sum_y / estimation
                )
                with np.errstate(
                    divide="ignore",
                    invalid="ignore",
                    over="ignore",
                ):
                    beta = centered_xy / centered_xx
                    alpha = (sum_y - beta * sum_x) / estimation
                    residual_variance = (
                        centered_yy
                        - centered_xy * centered_xy / centered_xx
                    ) / (estimation - 2)
                    correlation = centered_xy / np.sqrt(
                        centered_xx * centered_yy
                    )
                    score = -(
                        recent_sum_y[eligible]
                        - recent * alpha
                        - beta * recent_sum_x[eligible]
                    ) / np.sqrt(residual_variance * recent)

                valid_score = (
                    np.isfinite(centered_xx)
                    & (centered_xx > 0.0)
                    & np.isfinite(centered_yy)
                    & (centered_yy > 0.0)
                    & np.isfinite(residual_variance)
                    & (residual_variance > 0.0)
                    & np.isfinite(correlation)
                    & (correlation >= self.min_correlation)
                    & np.isfinite(score)
                )
                destination = eligible[valid_score]
                result[
                    output_row,
                    active_columns[destination],
                ] = score[valid_score]

            if output_row + 1 >= rows:
                continue

            estimation_out = output_row - total_window
            estimation_in = output_row - recent
            recent_out = estimation_in
            recent_in = output_row
            estimation_x_out = peer_return[estimation_out]
            estimation_y_out = log_return[estimation_out]
            estimation_x_in = peer_return[estimation_in]
            estimation_y_in = log_return[estimation_in]
            estimation_sum_x -= estimation_x_out
            estimation_sum_x += estimation_x_in
            estimation_sum_y -= estimation_y_out
            estimation_sum_y += estimation_y_in
            estimation_sum_xx -= estimation_x_out * estimation_x_out
            estimation_sum_xx += estimation_x_in * estimation_x_in
            estimation_sum_yy -= estimation_y_out * estimation_y_out
            estimation_sum_yy += estimation_y_in * estimation_y_in
            estimation_sum_xy -= estimation_x_out * estimation_y_out
            estimation_sum_xy += estimation_x_in * estimation_y_in

            recent_sum_x -= peer_return[recent_out]
            recent_sum_x += peer_return[recent_in]
            recent_sum_y -= log_return[recent_out]
            recent_sum_y += log_return[recent_in]
            full_window_valid -= pair_valid[estimation_out]
            full_window_valid += pair_valid[recent_in]
            transition_count -= industry_transition[
                output_row - total_window
            ]
            transition_count += industry_transition[output_row - 1]

        return result


class IndustryPeerReversal20Strict(_IndustryPeerReversalStrictBase):
    """20-day beta/correlation estimate and 5-day residual deviation."""

    estimation_window = 20
    deviation_window = 5
    hist_days = estimation_window + deviation_window


class IndustryPeerReversal60Strict(_IndustryPeerReversalStrictBase):
    """60-day beta/correlation estimate and 10-day residual deviation."""

    estimation_window = 60
    deviation_window = 10
    hist_days = estimation_window + deviation_window


class IndustryPeerReversal120Strict(_IndustryPeerReversalStrictBase):
    """120-day beta/correlation estimate and 20-day residual deviation."""

    estimation_window = 120
    deviation_window = 20
    hist_days = estimation_window + deviation_window
