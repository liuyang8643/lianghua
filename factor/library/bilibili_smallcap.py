"""Small-cap video adaptations using the canonical market-cap factor.

The caller selects ten stocks through WBR's existing planner. These research
definitions use its full PIT A-share pool and daily execution; the video's
index membership and final holding period are not fully disclosed. Market
cap uses T-open and lagged shares, rather than the video's completed close.
No IPO, industry or historical index-membership data are invented.
"""

from __future__ import annotations

import hashlib
import inspect
import json

import numpy as np

from factor import FactorDefinition, FactorMetadata, get_factor_definition


TRUE_MARKET_CAP_DEFINITION = get_factor_definition("TrueMarketCap")
VERSION = "bilibili-smallcap-v1-t-open-lagged-shares-completed-ma20"


class BiliSmallCapRisingMA20(TRUE_MARKET_CAP_DEFINITION.implementation):
    """Canonical negative market cap when completed MA20 rose strictly.

    ``total_share`` is lagged by the public FactorDefinition preprocessor,
    exactly as for production TrueMarketCap. At decision T, both MA20 values
    require the union of 21 completed positive closes in [T-21,T). Their
    difference is (close[T-1]-close[T-21])/20, so comparing the endpoints
    avoids copying price windows and subtracting large cumulative sums.
    Missing/nonpositive closes or a flat/falling mean yield NaN; this is an
    unavailable research signal, not an instruction to hold cash.
    """

    hist_days = 21

    def calc_batch(self, panel: dict[str, np.ndarray]) -> np.ndarray:
        score = super().calc_batch(panel)
        close = np.asarray(panel["close"])
        if close.ndim != 2 or score.shape != close.shape:
            raise ValueError("close and market-cap inputs must share date-by-stock axes")
        valid = np.isfinite(close) & (close > 0)
        count = np.concatenate(
            (np.zeros((1, close.shape[1]), dtype=np.int64), np.cumsum(valid, axis=0)),
            axis=0,
        )
        score[:self.hist_days] = np.nan
        if len(close) > self.hist_days:
            complete = (count[self.hist_days:] - count[:-self.hist_days])[:-1] == self.hist_days
            rising = close[self.hist_days - 1:-1] > close[:-self.hist_days]
            score[self.hist_days:][~(complete & rising)] = np.nan
        return score


_SOURCE = inspect.getsource(inspect.getmodule(BiliSmallCapRisingMA20))
_HASH = hashlib.sha256(json.dumps(
    {"source": _SOURCE, "version": VERSION, "market_cap": TRUE_MARKET_CAP_DEFINITION.metadata.as_dict()},
    sort_keys=True, separators=(",", ":"),
).encode()).hexdigest()
SMALLCAP_FACTOR_DEFINITIONS = (
    TRUE_MARKET_CAP_DEFINITION,
    FactorDefinition(
        metadata=FactorMetadata(
            name="BiliSmallCapRisingMA20", version=VERSION, hist_days=21,
            required_fields=(*TRUE_MARKET_CAP_DEFINITION.metadata.required_fields, "close"),
            implementation_hash=_HASH,
            lagged_fields=TRUE_MARKET_CAP_DEFINITION.metadata.lagged_fields,
        ),
        implementation=BiliSmallCapRisingMA20,
    ),
)


__all__ = ["BiliSmallCapRisingMA20", "TRUE_MARKET_CAP_DEFINITION", "SMALLCAP_FACTOR_DEFINITIONS"]
