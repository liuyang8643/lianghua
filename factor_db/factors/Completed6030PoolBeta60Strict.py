"""Strict 60-day beta to the completed 60/00/30-pool market return."""

from __future__ import annotations

import numpy as np


_WINDOW = 60
_MIN_MARKET_STOCKS = 30
_POOL_PREFIXES = ("60", "00", "30")


def _pool_mask(stock_codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(stock_codes)
    if codes.ndim != 1:
        raise ValueError("stock_codes must be one-dimensional")
    text = codes.astype("U", copy=False)
    return (
        np.char.startswith(text, _POOL_PREFIXES[0])
        | np.char.startswith(text, _POOL_PREFIXES[1])
        | np.char.startswith(text, _POOL_PREFIXES[2])
    )


def _official_return_row(
    close: np.ndarray,
    pre_close: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = (
        np.isfinite(close)
        & (close > 0.0)
        & np.isfinite(pre_close)
        & (pre_close > 0.0)
    )
    returns = np.zeros(close.shape, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(
            close,
            pre_close,
            out=returns,
            where=valid,
        )
        returns -= 1.0
    valid &= np.isfinite(returns)
    returns[~valid] = 0.0
    return returns, valid


def _compute_6030_market_returns(
    close: np.ndarray,
    pre_close: np.ndarray,
    stock_codes: np.ndarray,
) -> np.ndarray:
    """Return completed daily equal-weight 6030-pool returns.

    A day is finite only when at least 30 pool members have finite official
    returns.  Invalid members are omitted from that day's equal-weight mean;
    no value is filled or substituted.
    """
    if close.ndim != 2 or pre_close.ndim != 2:
        raise ValueError("close and preClose must be two-dimensional")
    if close.shape != pre_close.shape:
        raise ValueError("close and preClose must have matching shapes")
    pool = _pool_mask(stock_codes)
    if pool.shape != (close.shape[1],):
        raise ValueError("stock_codes must align with the price panel columns")

    pool_columns = np.flatnonzero(pool)
    market = np.full(close.shape[0], np.nan, dtype=np.float64)
    daily, valid = _official_return_row(
        close[:, pool_columns],
        pre_close[:, pool_columns],
    )
    count = np.count_nonzero(valid, axis=1)
    total = np.sum(daily, axis=1, dtype=np.float64)
    eligible = count >= _MIN_MARKET_STOCKS
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(total, count, out=market, where=eligible)
    market[~np.isfinite(market)] = np.nan
    return market


def _completed_window_sum(values: np.ndarray, window: int) -> np.ndarray:
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


def _completed_window_all(valid: np.ndarray, window: int) -> np.ndarray:
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


class Completed6030PoolBeta60Strict:
    """Prefer low beta to the completed 60/00/30-pool market.

    The market pool is fixed by ``stock_codes`` prefixes ``60``, ``00``, and
    ``30``.  On each completed day, the market return is the equal-weight mean
    of finite official returns ``close / preClose - 1`` inside that pool and
    is valid only when at least 30 pool members are valid.  A pool stock is
    included in its own market mean; no leave-one-out adjustment is made.

    Row ``T`` uses exactly ``[T-60, T)``.  All 60 stock returns and all 60
    market returns must be valid.  Beta is
    ``sum((r-r_mean)*(m-m_mean)) / sum((m-m_mean)**2)`` and the score is its
    negative.  A non-positive market centered-square sum invalidates the
    window.  There is no clipping, filling, shortened window, or fallback.

    Only close, official preClose, and stock_codes are read.  No current-row
    signal, open, gap, high, low, volume, amount, or external series is used.
    """

    hist_days = _WINDOW
    update_frequency = "daily"
    pre_ranked = False
    requires_full_history = False

    def calc_batch(self, panel: dict) -> np.ndarray:
        close = np.asarray(panel["close"], dtype=np.float64)
        pre_close = np.asarray(panel["preClose"], dtype=np.float64)
        stock_codes = np.asarray(panel["stock_codes"])
        if close.ndim != 2 or pre_close.ndim != 2:
            raise ValueError("close and preClose must be two-dimensional")
        if close.shape != pre_close.shape:
            raise ValueError("close and preClose must have matching shapes")
        if stock_codes.ndim != 1 or stock_codes.shape[0] != close.shape[1]:
            raise ValueError(
                "stock_codes must be one-dimensional and align with columns"
            )

        rows, stocks = close.shape
        result = np.full((rows, stocks), np.nan, dtype=np.float32)
        if rows <= _WINDOW:
            return result

        market_returns = _compute_6030_market_returns(
            close,
            pre_close,
            stock_codes,
        )
        daily_returns, daily_valid = _official_return_row(close, pre_close)
        return_offset = np.where(daily_valid[0], daily_returns[0], 0.0)
        daily_returns -= return_offset
        daily_returns[~daily_valid] = 0.0

        market_valid = np.isfinite(market_returns)
        market_offset = market_returns[0] if market_valid[0] else 0.0
        centered_market = np.where(
            market_valid,
            market_returns - market_offset,
            0.0,
        )
        market_column = centered_market[:, None]
        market_sum = _completed_window_sum(
            market_column.copy(),
            _WINDOW,
        )[:, 0]
        market_square_sum = _completed_window_sum(
            market_column * market_column,
            _WINDOW,
        )[:, 0]
        market_window_valid = _completed_window_all(
            market_valid[:, None],
            _WINDOW,
        )[:, 0]

        cross_product = daily_returns * market_column
        return_sum = _completed_window_sum(daily_returns, _WINDOW)
        cross_sum = _completed_window_sum(cross_product, _WINDOW)
        return_window_valid = _completed_window_all(
            daily_valid,
            _WINDOW,
        )

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            market_square_sum -= market_sum * market_sum / _WINDOW
            cross_sum -= return_sum * (market_sum / _WINDOW)[:, None]
            np.divide(
                -cross_sum,
                market_square_sum[:, None],
                out=cross_sum,
                where=market_square_sum[:, None] > 0.0,
            )

        valid = (
            return_window_valid
            & market_window_valid[:, None]
            & np.isfinite(market_square_sum)[:, None]
            & (market_square_sum > 0.0)[:, None]
            & np.isfinite(cross_sum)
        )
        result[_WINDOW:] = np.where(
            valid,
            cross_sum,
            np.nan,
        ).astype(np.float32)

        return result
