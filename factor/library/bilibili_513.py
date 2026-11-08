"""Research-only 513 variant: volume anomaly on the fifth bullish bar.

Source BV1y1ht6PEU1, 49--67s and 136--154s. The video shows daily
amplitude <=5% but does not disclose its denominator or adjustment code.
We explicitly use (high-low)/official preClose and raw OHLC. At decision T,
the anomaly is T-4, five bullish bars end there, and three confirmation
bars end at T-1. No T OHLC/volume enters the score. This is an event signal,
not a promise that the mandatory-full-investment portfolio trades only it.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
from factor.base import FactorDefinition, FactorMetadata

NAME = "Bili513Day5Signal"
VERSION = "bili513-day5-v1-raw-amplitude-official-preclose"
FIELDS = ("open", "high", "low", "close", "preClose", "volume")


class Bili513Day5Signal:
    hist_days = 8
    pre_ranked = False
    requires_full_history = False

    def calc_batch(self, panel: dict[str, np.ndarray]) -> np.ndarray:
        op, hi, lo, cl, pc, vol = (np.asarray(panel[k], dtype=np.float64) for k in FIELDS)
        if op.ndim != 2 or any(a.shape != op.shape for a in (hi, lo, cl, pc, vol)):
            raise ValueError("513 fields must be matching date-by-stock matrices")
        result = np.full(op.shape, np.nan)
        if len(op) <= self.hist_days:
            return result
        good = (np.isfinite(op) & (op > 0) & np.isfinite(hi) & np.isfinite(lo)
                & np.isfinite(cl) & (lo > 0) & (cl > 0)
                & (hi >= np.maximum(op, cl)) & (lo <= np.minimum(op, cl))
                & np.isfinite(vol) & (vol > 0))
        # Vectorized across dates and the complete stock axis; only eight
        # fixed offsets are visited, with no per-stock Python loop.
        n = len(op) - 8
        valid = np.ones((n, op.shape[1]), dtype=bool)
        signal = np.ones_like(valid)
        for offset in range(8):
            valid &= good[offset:offset+n]
            if offset < 5:
                signal &= cl[offset:offset+n] > op[offset:offset+n]
        anomaly = slice(4, 4+n)
        signal &= (hi[anomaly] - cl[anomaly]) > (cl[anomaly] - op[anomaly])
        signal &= vol[anomaly] > 2.0 * vol[3:3+n]
        for offset in (5, 6, 7):
            ix = slice(offset, offset+n)
            valid &= np.isfinite(pc[ix]) & (pc[ix] > 0)
            signal &= (hi[ix] - lo[ix]) <= .05 * pc[ix]
            signal &= cl[ix] >= op[anomaly]
            signal &= vol[ix] < vol[anomaly]
        result[8:] = np.where(valid, signal, np.nan)
        return result


DEFINITION = FactorDefinition(
    metadata=FactorMetadata(name=NAME, version=VERSION, hist_days=8,
        required_fields=FIELDS, score_semantics="binary",
        implementation_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
    implementation=Bili513Day5Signal,
)
