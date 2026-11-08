"""Common stock-universe filters shared by configured strategy factors."""

import numpy as np


MIN_RAW_PRICE = 2.0


class FilterST:
    """Exclude the runtime ST/*ST/delisting status mask."""

    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        return np.where(panel["st_mask"], np.nan, 1.0)


class FilterLowPrice:
    """Exclude stocks whose raw open price is below the configured floor."""

    hist_days = 0

    def calc_batch(self, panel: dict) -> np.ndarray:
        raw_open = panel["open"]
        return np.where(np.isfinite(raw_open) & (raw_open >= MIN_RAW_PRICE), 1.0, np.nan)
