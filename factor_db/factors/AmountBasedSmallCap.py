import numpy as np


class AmountBasedSmallCap:
    """小盘股因子 - 基于已完成日成交额近似市值。"""

    hist_days = 60

    def calc_batch(self, panel: dict) -> np.ndarray:
        from factor.library.completed_windows import iter_completed_cumulative_sums

        amount = np.asarray(panel["amount"])
        result = np.full(amount.shape, np.nan, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            for row, sums, counts, _ in iter_completed_cumulative_sums(amount, self.hist_days):
                average = sums / counts
                average /= 1e8
                score = 100 * np.exp(-(average / 5))
                result[row] = np.where(~np.isnan(average), score, np.nan)
        return result
