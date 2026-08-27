"""Strict prior-complete-month intraday trend consistency."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np


class CompletedPriorMonthIntradayTrendStrict:
    """Emit positive intraday Q from the previous complete natural month.

    Every decision row in natural month ``M`` receives the same score built
    from every market row ``d`` in ``M-1``:

    ``r[d] = log(close[d] / open[d])``
    ``Q = sum(r) / sqrt(n * sum(r**2))``.

    The previous month is valid for a stock only when every open/close pair is
    finite and strictly positive and the return energy is positive.  Only
    positive finite Q values are emitted; all other observations remain NaN,
    allowing the tactical sleeve to leave its slot in cash.  The first month
    present in a panel is never used as a history source because the panel may
    begin partway through that month.

    No current-month price field, preClose, high, low, volume, amount,
    total_share, ST state, overnight gap, or external data is read.
    """

    hist_days = 60
    update_frequency = "monthly"
    pre_ranked = False
    requires_full_history = False

    def calc_batch(self, panel: dict) -> np.ndarray:
        open_price = self._numeric_matrix(panel["open"], "open")
        close = self._numeric_matrix(panel["close"], "close")
        if open_price.shape != close.shape:
            raise ValueError("open and close must have matching shapes")

        rows, stocks = open_price.shape
        if rows == 0 or stocks == 0:
            raise ValueError(
                "open and close panels must contain at least one row and stock"
            )
        trade_dates = self._validated_trade_dates(panel["trade_dates"], rows)
        month_by_row = trade_dates.astype("datetime64[M]")
        month_starts = np.flatnonzero(
            np.r_[True, month_by_row[1:] != month_by_row[:-1]]
        )
        month_ends = np.r_[month_starts[1:], rows]
        month_numbers = month_by_row[month_starts].astype(np.int64)
        if len(month_numbers) > 1 and np.any(np.diff(month_numbers) != 1):
            raise ValueError(
                "trade_dates must span consecutive calendar months without gaps"
            )

        valid_daily = (
            np.isfinite(open_price)
            & (open_price > 0.0)
            & np.isfinite(close)
            & (close > 0.0)
        )
        daily_return = np.zeros(open_price.shape, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            np.divide(
                close,
                open_price,
                out=daily_return,
                where=valid_daily,
                dtype=np.float64,
            )
            np.log(daily_return, out=daily_return, where=valid_daily)
        valid_daily &= np.isfinite(daily_return)
        daily_return[~valid_daily] = 0.0

        monthly_sum = np.add.reduceat(
            daily_return,
            month_starts,
            axis=0,
        )
        np.square(daily_return, out=daily_return)
        monthly_energy = np.add.reduceat(
            daily_return,
            month_starts,
            axis=0,
        )
        monthly_valid = np.logical_and.reduceat(
            valid_daily,
            month_starts,
            axis=0,
        )
        month_lengths = (month_ends - month_starts).astype(np.float64)
        denominator = np.sqrt(
            month_lengths[:, None] * monthly_energy
        )
        monthly_q = np.zeros(monthly_sum.shape, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            np.divide(
                monthly_sum,
                denominator,
                out=monthly_q,
                where=monthly_energy > 0.0,
            )
        monthly_valid &= (
            (monthly_energy > 0.0)
            & np.isfinite(monthly_q)
            & (monthly_q > 0.0)
        )

        # A loader history slice may begin in the middle of its first month.
        # That boundary month can neither receive nor supply a strict score.
        monthly_valid[0] = False
        score_by_month = np.full(
            monthly_q.shape,
            np.nan,
            dtype=np.float32,
        )
        if len(month_starts) > 1:
            score_by_month[1:] = np.where(
                monthly_valid[:-1],
                monthly_q[:-1],
                np.nan,
            ).astype(np.float32)

        month_index_by_row = np.repeat(
            np.arange(len(month_starts)),
            month_ends - month_starts,
        )
        return score_by_month[month_index_by_row]

    @staticmethod
    def _numeric_matrix(values: object, name: str) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional matrix")
        if array.dtype.kind not in "iuf":
            raise ValueError(f"{name} must have a real numeric dtype")
        return array

    @staticmethod
    def _validated_trade_dates(values: object, rows: int) -> np.ndarray:
        raw = np.asarray(values)
        if raw.ndim != 1:
            raise ValueError("trade_dates must be one-dimensional")
        if len(raw) != rows:
            raise ValueError(
                "trade_dates length must match the price panel rows"
            )
        if raw.dtype.kind not in "MOSU":
            raise ValueError(
                "trade_dates must contain datetime64 or date-like values"
            )
        if raw.dtype.kind == "O" and any(
            not isinstance(
                value,
                (date, datetime, np.datetime64, str, np.str_),
            )
            for value in raw
        ):
            raise ValueError(
                "trade_dates must contain datetime64 or date-like values"
            )
        try:
            dates = raw.astype("datetime64[D]")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "trade_dates must contain valid calendar dates"
            ) from exc
        if np.isnat(dates).any():
            raise ValueError("trade_dates must not contain NaT")
        if len(dates) > 1 and np.any(dates[1:] <= dates[:-1]):
            raise ValueError("trade_dates must be strictly increasing")
        return dates
