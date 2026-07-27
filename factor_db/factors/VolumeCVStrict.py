"""Strict completed-volume coefficient of variation."""

from __future__ import annotations

import numpy as np


class VolumeCVStrict:
    """Prefer stable volume using exactly 20 completed, finite observations."""

    hist_days = 20
    pre_ranked = False
    requires_full_history = False

    def calc_batch(self, panel: dict) -> np.ndarray:
        volume = np.asarray(panel["volume"], dtype=np.float64)
        rows, stocks = volume.shape
        result = np.full((rows, stocks), np.nan, dtype=np.float32)
        window = self.hist_days
        if rows <= window:
            return result

        finite = np.isfinite(volume)
        values = np.where(finite, volume, 0.0)
        counts = np.sum(finite[:window], axis=0, dtype=np.int32)
        sums = np.sum(values[:window], axis=0, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            sums_sq = np.sum(
                values[:window] * values[:window], axis=0, dtype=np.float64,
            )

        # A fixed-width rolling state avoids three full-panel prefix arrays.
        # Rebase periodically so repeated add/subtract updates cannot accumulate
        # enough floating-point drift to change a cross-sectional rank.
        rebase_interval = 256
        for row in range(window, rows):
            if row > window and row % rebase_interval == 0:
                start = row - window
                history = values[start:row]
                counts = np.sum(finite[start:row], axis=0, dtype=np.int32)
                sums = np.sum(history, axis=0, dtype=np.float64)
                with np.errstate(over="ignore", invalid="ignore"):
                    sums_sq = np.sum(
                        history * history, axis=0, dtype=np.float64,
                    )

            # Row T uses volume[T-window:T], never volume[T].
            means = sums / window
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                variances = np.maximum(sums_sq / window - means * means, 0.0)
                cv = np.sqrt(variances) / means
            valid = (
                (counts == window)
                & np.isfinite(means)
                & (means > 0.0)
                & np.isfinite(cv)
            )
            result[row, valid] = -cv[valid]

            if row + 1 < rows:
                outgoing = row - window
                counts += finite[row]
                counts -= finite[outgoing]
                sums += values[row]
                sums -= values[outgoing]
                with np.errstate(over="ignore", invalid="ignore"):
                    sums_sq += values[row] * values[row]
                    sums_sq -= values[outgoing] * values[outgoing]

        return result
