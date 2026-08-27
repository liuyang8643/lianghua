"""Time-invariant cross-sectional market-cap floors for non-small-cap styles."""

from __future__ import annotations

import numpy as np


class _FilterMarketCapTopFraction:
    hist_days = 0
    top_fraction: float

    def calc_batch(self, panel: dict) -> np.ndarray:
        pre_close = np.asarray(panel["preClose"], dtype=np.float64)
        total_share = np.asarray(panel["total_share"], dtype=np.float64)
        stock_codes = np.asarray(panel["stock_codes"]).astype("U16")
        pool = (
            np.char.startswith(stock_codes, "60")
            | np.char.startswith(stock_codes, "00")
            | np.char.startswith(stock_codes, "30")
        )
        with np.errstate(invalid="ignore", over="ignore"):
            market_cap = pre_close * total_share
        valid = (
            np.isfinite(pre_close)
            & (pre_close > 0.0)
            & np.isfinite(total_share)
            & (total_share > 0.0)
            & np.isfinite(market_cap)
            & pool[None, :]
        )
        values = np.where(valid, market_cap, np.nan)
        with np.errstate(invalid="ignore"):
            threshold = np.nanquantile(
                values,
                1.0 - self.top_fraction,
                axis=1,
            )
        allowed = valid & (market_cap >= threshold[:, None])
        return np.where(allowed, 1.0, np.nan).astype(np.float32)


class FilterMarketCapTop50Pct(_FilterMarketCapTopFraction):
    """Exclude the smaller half of the investable 60/00/30 universe."""

    top_fraction = 0.50


class FilterMarketCapTop40Pct(_FilterMarketCapTopFraction):
    """Keep only the largest 40% of the investable 60/00/30 universe."""

    top_fraction = 0.40


class FilterMarketCapTop30Pct(_FilterMarketCapTopFraction):
    """Keep only the largest 30% of the investable 60/00/30 universe."""

    top_fraction = 0.30


class FilterMarketCapTop20Pct(_FilterMarketCapTopFraction):
    """Keep only the largest 20% of the investable 60/00/30 universe."""

    top_fraction = 0.20


class FilterMarketCapTop10Pct(_FilterMarketCapTopFraction):
    """Keep only the largest 10% of the investable 60/00/30 universe."""

    top_fraction = 0.10
