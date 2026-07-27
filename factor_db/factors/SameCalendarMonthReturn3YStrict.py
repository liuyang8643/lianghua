"""Strict same-calendar-month return at annual lags one through three."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np


_ANNUAL_LAGS = np.array((12, 24, 36), dtype=np.int64)


class SameCalendarMonthReturn3YStrict:
    """Prefer stocks with strong returns in the same month of prior years.

    At the first market row of natural month ``M``, each historical monthly
    return compounds every official gross return ``close[d] / preClose[d]``
    in exactly ``M-12``, ``M-24``, and ``M-36``.  The score is their
    arithmetic mean and is frozen for all rows in ``M``.  Every row of all
    three historical months must have finite, strictly positive inputs and an
    official gross return with the same properties.

    The first natural month present in the input panel is deliberately never
    considered complete because a production history slice can begin partway
    through that month.  Current-month ``open`` and ``st_mask`` are legality
    gates only; current-month HLCVA and ``preClose`` never enter its score.
    """

    hist_days = 800
    pre_ranked = False
    requires_full_history = False

    def calc_batch(self, panel: dict) -> np.ndarray:
        open_ = self._numeric_matrix(panel["open"], "open")
        close = self._numeric_matrix(panel["close"], "close")
        pre_close = self._numeric_matrix(panel["preClose"], "preClose")
        st_mask = np.asarray(panel["st_mask"])

        if not (
            open_.shape == close.shape == pre_close.shape == st_mask.shape
        ):
            raise ValueError(
                "open, close, preClose, and st_mask must have matching shapes"
            )
        if st_mask.dtype.kind != "b":
            raise ValueError("st_mask must be a boolean matrix")

        rows, stocks = close.shape
        if rows == 0 or stocks == 0:
            raise ValueError("price panels must contain at least one row and stock")

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

        close_64 = np.asarray(close, dtype=np.float64)
        pre_close_64 = np.asarray(pre_close, dtype=np.float64)
        valid_daily = (
            np.isfinite(close_64)
            & (close_64 > 0.0)
            & np.isfinite(pre_close_64)
            & (pre_close_64 > 0.0)
        )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            daily_log_gross = np.divide(
                close_64,
                pre_close_64,
                dtype=np.float64,
            )
        valid_daily &= (
            np.isfinite(daily_log_gross) & (daily_log_gross > 0.0)
        )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            np.log(
                daily_log_gross,
                out=daily_log_gross,
                where=valid_daily,
            )
        valid_daily &= np.isfinite(daily_log_gross)
        daily_log_gross[~valid_daily] = 0.0

        monthly_log_gross = np.add.reduceat(
            daily_log_gross,
            month_starts,
            axis=0,
        )
        valid_month = np.logical_and.reduceat(
            valid_daily,
            month_starts,
            axis=0,
        )
        with np.errstate(invalid="ignore", over="ignore", under="ignore"):
            monthly_returns = np.expm1(monthly_log_gross)
        valid_month &= np.isfinite(monthly_returns)
        monthly_returns[~valid_month] = np.nan

        # A max-lookback slice does not prove that its boundary month contains
        # every market row, even when its first observed date looks month-like.
        valid_month[0] = False
        monthly_returns[0] = np.nan

        month_count = len(month_starts)
        month_scores = np.full(
            (month_count, stocks),
            np.nan,
            dtype=np.float64,
        )
        target_months = np.arange(37, month_count, dtype=np.int64)
        if len(target_months):
            lag_indices = target_months[:, None] - _ANNUAL_LAGS[None, :]
            lag_returns = monthly_returns[lag_indices]
            valid_history = np.all(valid_month[lag_indices], axis=1)
            # Divide before summing so the arithmetic mean cannot overflow
            # merely because three individually finite returns are very large.
            with np.errstate(invalid="ignore", over="ignore"):
                scores = np.sum(
                    lag_returns / 3.0,
                    axis=1,
                    dtype=np.float64,
                )
            valid_score = valid_history & np.isfinite(scores)
            month_scores[target_months] = np.where(
                valid_score,
                scores,
                np.nan,
            )

        result = np.full((rows, stocks), np.nan, dtype=np.float32)
        for month_index, (start, end) in enumerate(
            zip(month_starts, month_ends)
        ):
            score = month_scores[month_index]
            if not np.any(np.isfinite(score)):
                continue
            start = int(start)
            end = int(end)
            legal = (
                np.isfinite(open_[start:end])
                & (open_[start:end] >= 2.0)
                & ~st_mask[start:end]
            )
            valid = legal & np.isfinite(score)[None, :]
            result[start:end] = np.where(valid, score[None, :], np.nan)

        return result

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
            raise ValueError("trade_dates length must match the price panel rows")
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
