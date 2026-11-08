"""Completed-window reductions over date-by-stock matrices.

The cumulative iterator reads immutable raw inputs. Prefix/suffix reductions
either read immutable inputs with explicit finite counts or reduce disposable
workspaces in place for the strict research-factor contracts.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False, parallel=False, error_model="numpy")
def _completed_finite_sum_kernel(values, window, skip, required):
    """Same reverse suffix/forward prefix additions, fused bounded workspace."""
    rows, stocks = values.shape
    result = np.full((rows, stocks), np.nan, dtype=np.float64)
    if rows <= window + skip:
        return result
    suffix = np.empty((window, stocks), dtype=values.dtype)
    suffix_count = np.empty((window, stocks), dtype=np.int32)
    prefix = np.empty(stocks, dtype=np.float64)
    prefix_count = np.empty(stocks, dtype=np.int32)
    for end in range(window, rows - skip, window):
        length = min(window, rows - skip - end)
        for stock in range(stocks):
            value = values[end - 1, stock]
            valid = np.isfinite(value)
            suffix[window - 1, stock] = value if valid else 0.0
            suffix_count[window - 1, stock] = int(valid)
        for offset in range(window - 2, -1, -1):
            for stock in range(stocks):
                value = values[end - window + offset, stock]
                valid = np.isfinite(value)
                # Store into source dtype each step, matching np.cumsum's
                # float32 or float64 suffix accumulator, without fast-math.
                suffix[offset, stock] = suffix[offset + 1, stock] + (value if valid else 0.0)
                suffix_count[offset, stock] = suffix_count[offset + 1, stock] + int(valid)
        for stock in range(stocks):
            prefix[stock] = 0.0
            prefix_count[stock] = 0
            if suffix_count[0, stock] >= required:
                result[end + skip, stock] = suffix[0, stock] + prefix[stock]
        for offset in range(1, length):
            for stock in range(stocks):
                value = values[end + offset - 1, stock]
                valid = np.isfinite(value)
                if offset == 1:
                    prefix[stock] = value if valid else 0.0
                else:
                    prefix[stock] += value if valid else 0.0
                prefix_count[stock] += int(valid)
                if suffix_count[offset, stock] + prefix_count[stock] >= required:
                    result[end + skip + offset, stock] = suffix[offset, stock] + prefix[stock]
    return result


def completed_finite_window_sum(
    values: np.ndarray, window: int, skip: int = 0, *, min_count: int | None = None,
) -> np.ndarray:
    """Completed finite windows; native float32/64 retain their sum precision.

    Suffix accumulation uses the source dtype and prefix accumulation float64,
    preserving the previous NumPy arithmetic. Unsupported dtypes are rejected
    rather than silently changing their accumulator precision.
    """
    if values.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("completed finite windows require native float32 or float64 inputs")
    return _completed_finite_sum_kernel(values, window, skip, window if min_count is None else min_count)


def iter_completed_cumulative_sums(
    values: np.ndarray, window: int, *, include_squares: bool = False,
):
    """Yield T-1 cumulative differences with bounded stock-vector workspaces.

    This deliberately retains the legacy cumulative arithmetic, including its
    cancellation and infinity propagation. The ring stores earlier cumulative
    values, not independent window sums. Float32 inputs retain float32 sums
    and squares; observation counts have the original float64 result dtype.
    Rows before ``window`` yield the available prefix, allowing each consumer
    to impose its existing warm-up rule. The first row is unavailable.
    """
    values = np.asarray(values)
    rows, stocks = values.shape
    if rows == 0:
        raise IndexError("completed factor history must contain at least one row")
    cumulative = np.zeros(stocks, dtype=values.dtype)
    count = np.zeros(stocks, dtype=np.float64)
    prior_sum = np.zeros((window, stocks), dtype=values.dtype)
    prior_count = np.zeros((window, stocks), dtype=np.float64)
    if include_squares:
        square_sum = np.zeros(stocks, dtype=values.dtype)
        prior_square = np.zeros((window, stocks), dtype=values.dtype)
    for row in range(1, rows):
        current = values[row - 1]
        valid = ~np.isnan(current)
        cumulative += np.where(valid, current, 0.0)
        count += valid
        slot = row % window
        if include_squares:
            square_sum += np.where(valid, current * current, 0.0)
            squares = square_sum - prior_square[slot]
            prior_square[slot] = square_sum
        else:
            squares = None
        sums = cumulative - prior_sum[slot]
        counts = count - prior_count[slot]
        prior_sum[slot] = cumulative
        prior_count[slot] = count
        yield row, sums, counts, squares



def completed_window_sum(values: np.ndarray, window: int) -> np.ndarray:
    """Sum independent completed windows through block suffixes/prefixes."""
    rows, stocks = values.shape
    output = np.empty((rows - window, stocks), dtype=np.float64)
    full_blocks = rows // window
    full_rows = full_blocks * window
    blocks = values[:full_rows].reshape(full_blocks, window, stocks)
    complete_output_rows = (full_blocks - 1) * window

    if complete_output_rows:
        complete = output[:complete_output_rows].reshape(
            full_blocks - 1,
            window,
            stocks,
        )
        np.cumsum(
            blocks[:-1, ::-1],
            axis=1,
            dtype=np.float64,
            out=complete[:, ::-1],
        )

    remainder = rows - full_rows
    if remainder:
        last_suffix = np.empty((window, stocks), dtype=np.float64)
        np.cumsum(
            blocks[-1, ::-1],
            axis=0,
            dtype=np.float64,
            out=last_suffix[::-1],
        )
        output[complete_output_rows:] = last_suffix[:remainder]

    if full_blocks > 1:
        np.cumsum(
            blocks[1:],
            axis=1,
            dtype=np.float64,
            out=blocks[1:],
        )
        complete = output[:complete_output_rows].reshape(
            full_blocks - 1,
            window,
            stocks,
        )
        complete[:, 1:] += blocks[1:, :-1]

    if remainder:
        tail = values[full_rows:]
        np.cumsum(tail, axis=0, dtype=np.float64, out=tail)
        output[complete_output_rows + 1 :] += tail[:-1]

    return output


def completed_window_all(valid: np.ndarray, window: int) -> np.ndarray:
    """Apply strict missing propagation to independent completed windows."""
    rows, stocks = valid.shape
    output = np.empty((rows - window, stocks), dtype=bool)
    full_blocks = rows // window
    full_rows = full_blocks * window
    blocks = valid[:full_rows].reshape(full_blocks, window, stocks)
    complete_output_rows = (full_blocks - 1) * window

    if complete_output_rows:
        complete = output[:complete_output_rows].reshape(
            full_blocks - 1,
            window,
            stocks,
        )
        np.logical_and.accumulate(
            blocks[:-1, ::-1],
            axis=1,
            out=complete[:, ::-1],
        )

    remainder = rows - full_rows
    if remainder:
        last_suffix = np.empty((window, stocks), dtype=bool)
        np.logical_and.accumulate(
            blocks[-1, ::-1],
            axis=0,
            out=last_suffix[::-1],
        )
        output[complete_output_rows:] = last_suffix[:remainder]

    if full_blocks > 1:
        np.logical_and.accumulate(
            blocks[1:],
            axis=1,
            out=blocks[1:],
        )
        complete = output[:complete_output_rows].reshape(
            full_blocks - 1,
            window,
            stocks,
        )
        complete[:, 1:] &= blocks[1:, :-1]

    if remainder:
        tail = valid[full_rows:]
        np.logical_and.accumulate(tail, axis=0, out=tail)
        output[complete_output_rows + 1 :] &= tail[:-1]

    return output
