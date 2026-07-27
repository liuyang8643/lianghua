"""Strict completed-volume contraction over recent and prior windows."""

from __future__ import annotations

import numpy as np


class VolumeContraction5v15Strict:
    """Prefer stocks whose completed five-day volume has contracted.

    Row ``T`` is scored as

    ``-log(mean(volume[T-5:T]) / mean(volume[T-20:T-5]))``.

    All 20 completed observations must be finite and strictly positive.  The
    current open and ST state are legality gates only; no T-day HLCVA value is
    part of the signal.
    """

    hist_days = 20
    pre_ranked = False
    requires_full_history = False

    def calc_batch(self, panel: dict) -> np.ndarray:
        volume = np.asarray(panel["volume"], dtype=np.float64)
        open_ = np.asarray(panel["open"], dtype=np.float64)
        st_mask = np.asarray(panel["st_mask"], dtype=bool)
        if volume.ndim != 2:
            raise ValueError("volume panel must be two-dimensional")
        if not (open_.shape == volume.shape == st_mask.shape):
            raise ValueError("volume, open, and st_mask must have matching shapes")

        rows, stocks = volume.shape
        result = np.full((rows, stocks), np.nan, dtype=np.float32)
        window = self.hist_days
        recent_window = 5
        prior_window = window - recent_window
        if rows <= window:
            return result

        valid_daily = np.isfinite(volume) & (volume > 0.0)
        safe_volume = np.where(valid_daily, volume, 0.0)

        prior_count = np.sum(
            valid_daily[:prior_window], axis=0, dtype=np.int32,
        )
        recent_count = np.sum(
            valid_daily[prior_window:window], axis=0, dtype=np.int32,
        )
        prior_sum = np.sum(
            safe_volume[:prior_window], axis=0, dtype=np.float64,
        )
        recent_sum = np.sum(
            safe_volume[prior_window:window], axis=0, dtype=np.float64,
        )

        # Periodic exact rebasing prevents accumulated add/subtract drift from
        # changing close cross-sectional ranks on long runtime panels.
        rebase_interval = 256
        for row in range(window, rows):
            if row > window and row % rebase_interval == 0:
                start = row - window
                split = row - recent_window
                prior_count = np.sum(
                    valid_daily[start:split], axis=0, dtype=np.int32,
                )
                recent_count = np.sum(
                    valid_daily[split:row], axis=0, dtype=np.int32,
                )
                prior_sum = np.sum(
                    safe_volume[start:split], axis=0, dtype=np.float64,
                )
                recent_sum = np.sum(
                    safe_volume[split:row], axis=0, dtype=np.float64,
                )

            prior_mean = prior_sum / prior_window
            recent_mean = recent_sum / recent_window
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                score = -np.log(recent_mean / prior_mean)
            legal = (
                np.isfinite(open_[row])
                & (open_[row] >= 2.0)
                & ~st_mask[row]
            )
            valid = (
                legal
                & (prior_count == prior_window)
                & (recent_count == recent_window)
                & np.isfinite(prior_mean)
                & (prior_mean > 0.0)
                & np.isfinite(recent_mean)
                & (recent_mean > 0.0)
                & np.isfinite(score)
            )
            result[row, valid] = score[valid]

            if row + 1 < rows:
                outgoing = row - window
                boundary = row - recent_window
                incoming = row
                prior_count += valid_daily[boundary]
                prior_count -= valid_daily[outgoing]
                recent_count += valid_daily[incoming]
                recent_count -= valid_daily[boundary]
                prior_sum += safe_volume[boundary]
                prior_sum -= safe_volume[outgoing]
                recent_sum += safe_volume[incoming]
                recent_sum -= safe_volume[boundary]

        return result
