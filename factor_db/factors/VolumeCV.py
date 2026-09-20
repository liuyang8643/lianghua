import numpy as np


class VolumeCV:
    """成交量变异系数 — std(volume)/mean(volume)，仅消费已完成日。"""

    hist_days = 20

    def calc_batch(self, panel: dict) -> np.ndarray:
        from factor.library.completed_windows import iter_completed_cumulative_sums

        volume = np.asarray(panel["volume"])
        result = np.full(volume.shape, np.nan, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            for row, sums, counts, squares in iter_completed_cumulative_sums(
                volume, self.hist_days, include_squares=True,
            ):
                if row < self.hist_days:
                    continue
                mean = sums / counts
                mean_sq = squares / counts
                variance = mean_sq - mean * mean
                cv = np.sqrt(np.maximum(variance, 0.0)) / mean
                result[row] = np.where(~np.isnan(cv), -cv, np.nan)
        return result
