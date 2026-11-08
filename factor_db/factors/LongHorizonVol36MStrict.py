"""Strict 36-complete-calendar-month low-volatility factor."""

from __future__ import annotations

import numpy as np


_HISTORY_MONTHS = 36


class LongHorizonVol36MStrict:
    """Prefer stocks with low volatility over 36 completed calendar months.

    For every date in calendar month ``M``, the signal is the negative
    population standard deviation of the 36 monthly official returns from
    ``M-36`` through ``M-1``.  A monthly return compounds
    ``close[d] / preClose[d]`` over every market row in that month.  One
    missing or invalid daily observation invalidates that stock-month, and all
    36 stock-months must be valid.  The first month present in the panel is
    deliberately unusable because a history slice may begin part-way through
    it.  Current ``open`` and ``st_mask`` values are legality gates only.
    """

    # 900 trading rows conservatively cover 37 calendar months plus a
    # potentially partial boundary month under the mainland trading calendar.
    hist_days = 900
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
        months = month_by_row[month_starts]
        month_numbers = months.astype(np.int64)
        if len(month_numbers) > 1 and np.any(np.diff(month_numbers) != 1):
            raise ValueError(
                "trade_dates must span consecutive calendar months without gaps"
            )

        month_count = len(months)
        monthly_returns = np.full(
            (month_count, stocks),
            np.nan,
            dtype=np.float64,
        )

        # Month zero can be a partial month introduced by max-lookback slicing.
        # Treating it as invalid is what makes all later 36-month windows safe.
        for month_index in range(1, month_count):
            start = int(month_starts[month_index])
            end = int(month_ends[month_index])
            close_month = np.asarray(close[start:end], dtype=np.float64)
            pre_close_month = np.asarray(pre_close[start:end], dtype=np.float64)
            valid_input = (
                np.isfinite(close_month)
                & (close_month > 0.0)
                & np.isfinite(pre_close_month)
                & (pre_close_month > 0.0)
            )
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                log_gross = np.log(close_month / pre_close_month)
            valid_daily = valid_input & np.isfinite(log_gross)
            safe_log_gross = np.where(valid_daily, log_gross, 0.0)
            log_month_return = np.sum(
                safe_log_gross,
                axis=0,
                dtype=np.float64,
            )
            with np.errstate(invalid="ignore", over="ignore"):
                month_return = np.expm1(log_month_return)
            valid_month = (
                np.all(valid_daily, axis=0)
                & np.isfinite(month_return)
            )
            monthly_returns[month_index, valid_month] = month_return[valid_month]

        result = np.full((rows, stocks), np.nan, dtype=np.float32)
        first_signal_month = _HISTORY_MONTHS + 1
        for month_index in range(first_signal_month, month_count):
            history = monthly_returns[
                month_index - _HISTORY_MONTHS : month_index
            ]
            valid_history = np.all(np.isfinite(history), axis=0)
            score = np.full(stocks, np.nan, dtype=np.float64)
            if np.any(valid_history):
                with np.errstate(invalid="ignore", over="ignore"):
                    score[valid_history] = -np.std(
                        history[:, valid_history],
                        axis=0,
                        dtype=np.float64,
                    )
                score[~np.isfinite(score)] = np.nan

            start = int(month_starts[month_index])
            end = int(month_ends[month_index])
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
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"{name} must be numeric")
        return array

    @staticmethod
    def _validated_trade_dates(values: object, rows: int) -> np.ndarray:
        raw = np.asarray(values)
        if raw.ndim != 1:
            raise ValueError("trade_dates must be one-dimensional")
        if len(raw) != rows:
            raise ValueError("trade_dates length must match the price panel rows")
        try:
            dates = raw.astype("datetime64[D]")
        except (TypeError, ValueError) as exc:
            raise ValueError("trade_dates must contain valid calendar dates") from exc
        if np.isnat(dates).any():
            raise ValueError("trade_dates must not contain NaT")
        if len(dates) > 1 and np.any(dates[1:] <= dates[:-1]):
            raise ValueError("trade_dates must be strictly increasing")
        return dates
