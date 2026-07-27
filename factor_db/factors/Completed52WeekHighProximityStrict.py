"""Strict proximity to the completed 52-week official-return high."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np


_COMPLETED_DAYS = 252


class Completed52WeekHighProximityStrict:
    """Prefer stocks nearest their completed 252-day official-return high.

    At the first market row ``s`` of each calendar month, the factor builds a
    wealth path from one using exactly the completed official gross returns
    ``close[d] / preClose[d]`` for ``d`` in ``[s-252, s)``.  Its score is the
    ending wealth divided by the maximum of one and every completed-day path
    wealth.  The score is fixed for that whole month and higher is preferred.

    Every one of the 252 gross returns must be finite and strictly positive.
    Current-month HLCVA and ``preClose`` values are never signal inputs;
    current ``open`` and ``st_mask`` values are legality gates only.
    """

    # The extra rows beyond the exact 252-day signal window only ensure that a
    # max-lookback slice can reach the first market row of the target month.
    hist_days = 280
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

        # Keep the official gross-return definition explicit.  Invalid ratios
        # become additive zeros solely so cumulative calculation can continue;
        # valid_history below still strictly poisons every containing window.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            log_wealth = np.divide(close, pre_close, dtype=np.float64)
        valid_daily = (
            np.isfinite(close)
            & (close > 0.0)
            & np.isfinite(pre_close)
            & (pre_close > 0.0)
            & np.isfinite(log_wealth)
            & (log_wealth > 0.0)
        )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            np.log(log_wealth, out=log_wealth, where=valid_daily)
        valid_daily &= np.isfinite(log_wealth)
        log_wealth[~valid_daily] = 0.0

        # In-place cumulative log wealth makes every monthly 252-day path a
        # pair of slices, without constructing a three-dimensional panel.
        np.cumsum(log_wealth, axis=0, dtype=np.float64, out=log_wealth)

        result = np.full((rows, stocks), np.nan, dtype=np.float32)
        for month_start, month_end in zip(month_starts, month_ends):
            end = int(month_start)
            if end < _COMPLETED_DAYS:
                continue
            start = end - _COMPLETED_DAYS

            valid_history = np.all(valid_daily[start:end], axis=0)
            if start == 0:
                starting_log_wealth = np.zeros(stocks, dtype=np.float64)
            else:
                starting_log_wealth = log_wealth[start - 1]
            ending_log_wealth = log_wealth[end - 1]
            peak_log_wealth = np.maximum(
                starting_log_wealth,
                np.max(log_wealth[start:end], axis=0),
            )
            with np.errstate(invalid="ignore", over="ignore", under="ignore"):
                score = np.exp(ending_log_wealth - peak_log_wealth)
            valid_score = valid_history & np.isfinite(score)

            row_start = int(month_start)
            row_end = int(month_end)
            legal = (
                np.isfinite(open_[row_start:row_end])
                & (open_[row_start:row_end] >= 2.0)
                & ~st_mask[row_start:row_end]
            )
            valid = legal & valid_score[None, :]
            result[row_start:row_end] = np.where(
                valid,
                score[None, :],
                np.nan,
            )

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
