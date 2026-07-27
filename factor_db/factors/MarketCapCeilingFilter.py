"""Absolute market-cap ceilings for stable micro-cap style exposure."""

from __future__ import annotations

import numpy as np


class _MarketCapCeilingFilter:
    """Return a strict boolean-style mask using market cap known at Open[T]."""

    hist_days = 0
    max_market_cap_yi: float

    def calc_batch(self, panel: dict) -> np.ndarray:
        open_price = np.asarray(panel["open"], dtype=np.float64)
        total_share_wan = np.asarray(panel["total_share"], dtype=np.float64)
        valid = (
            np.isfinite(open_price)
            & (open_price > 0.0)
            & np.isfinite(total_share_wan)
            & (total_share_wan > 0.0)
        )
        # total_share is stored in 10,000 shares.  Yuan * 10,000 shares /
        # 10,000 converts the result to 100 million yuan (亿元).
        market_cap_yi = open_price * total_share_wan / 1e4
        allowed = valid & (market_cap_yi <= self.max_market_cap_yi)
        return np.where(allowed, 1.0, np.nan).astype(np.float32)


class FilterMarketCapMax25Yi(_MarketCapCeilingFilter):
    max_market_cap_yi = 25.0


class FilterMarketCapMax28Yi(_MarketCapCeilingFilter):
    max_market_cap_yi = 28.0


class FilterMarketCapMax30Yi(_MarketCapCeilingFilter):
    max_market_cap_yi = 30.0


class FilterMarketCapMax32Yi(_MarketCapCeilingFilter):
    max_market_cap_yi = 32.0


class FilterMarketCapMax35Yi(_MarketCapCeilingFilter):
    max_market_cap_yi = 35.0
