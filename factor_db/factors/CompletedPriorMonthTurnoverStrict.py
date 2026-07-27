"""Strict prior-complete-natural-month share-turnover factor."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np


class CompletedPriorMonthTurnoverStrict:
    """Prefer low average share turnover in the prior complete month.

    Every row in natural month ``M`` receives the negative arithmetic mean
    of ``volume[d] / (100 * total_share[d])`` over every market row in
    ``M-1``.  Both inputs must be finite and strictly positive on every row
    for that stock; one invalid observation poisons the stock's entire score
    for ``M``.  The score is frozen throughout ``M``.

    The first natural month present in a panel is deliberately invalid
    because a loader history slice may begin partway through that month.
    No current-month market field is consumed by the signal.
    """

    hist_days = 60
    pre_ranked = False
    requires_full_history = False

    def calc_batch(self, panel: dict) -> np.ndarray:
        volume = self._numeric_matrix(panel["volume"], "volume")
        total_share = self._numeric_matrix(
            panel["total_share"],
            "total_share",
        )
        if volume.shape != total_share.shape:
            raise ValueError("volume and total_share must have matching shapes")

        rows, stocks = volume.shape
        if rows == 0 or stocks == 0:
            raise ValueError(
                "turnover panels must contain at least one row and stock"
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
            np.isfinite(volume)
            & (volume > 0)
            & np.isfinite(total_share)
            & (total_share > 0)
        )
        daily_turnover = np.zeros(volume.shape, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            np.divide(
                volume,
                total_share,
                out=daily_turnover,
                where=valid_daily,
                dtype=np.float64,
            )
            daily_turnover /= 100.0

        daily_turnover[~valid_daily] = 0.0

        monthly_sums = np.add.reduceat(
            daily_turnover,
            month_starts,
            axis=0,
        )
        monthly_valid = np.logical_and.reduceat(
            valid_daily,
            month_starts,
            axis=0,
        )
        month_lengths = (month_ends - month_starts).astype(np.float64)
        with np.errstate(invalid="ignore", over="ignore"):
            monthly_scores = -monthly_sums / month_lengths[:, None]
        monthly_valid &= np.isfinite(monthly_scores)

        # A boundary month can be partial even when its first observed row
        # resembles a normal month opening.  It can never supply M-1.
        monthly_valid[0] = False

        score_by_month = np.full(
            monthly_scores.shape,
            np.nan,
            dtype=np.float32,
        )
        prior_valid = monthly_valid[:-1]
        score_by_month[1:] = np.where(
            prior_valid,
            monthly_scores[:-1],
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
                "trade_dates length must match the turnover panel rows"
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
