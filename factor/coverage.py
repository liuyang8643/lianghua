"""Pure, PIT-member factor coverage audit for an already opened split."""

from __future__ import annotations

import numpy as np

from offline_data import RuntimeSlice
from .base import FactorBatch


def factor_coverage(runtime: RuntimeSlice, factors: FactorBatch) -> dict[str, object]:
    """Count missing outputs, including warmup/gaps, without reading other splits.

    Suspended members remain in the denominator. Future listings and stocks
    already delisted do not. Coverage is availability, not proof of PIT quality.
    """
    if runtime.stock_codes != factors.stock_codes or not np.array_equal(
        runtime.trade_dates, factors.trade_dates
    ):
        raise ValueError("coverage runtime/factor axes differ")
    start, stop = factors.decision_start, factors.decision_stop
    dates = factors.trade_dates[start:stop]
    member = (runtime.field("listing_age")[start:stop] >= 0) & ~runtime.field(
        "delisted_mask"
    )[start:stop]
    counts = member.sum(axis=1, dtype=np.int64)
    year_axis = dates.astype("datetime64[Y]").astype(str)
    entries: dict[str, object] = {}
    for index, metadata in enumerate(factors.factor_metadata):
        valid = factors.validity[start:stop, index] & member
        available = valid.sum(axis=1, dtype=np.int64)
        ratio = np.divide(available, counts, out=np.zeros(len(dates)), where=counts > 0)
        observed = np.flatnonzero(available > 0)
        absent = (available == 0) & (counts > 0)
        boundaries = np.diff(np.r_[False, absent, False].astype(np.int8))
        intervals = [
            {"start": str(dates[left]), "end": str(dates[right - 1]), "trading_rows": int(right - left)}
            for left, right in zip(np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1), strict=True)
        ]
        by_year = {}
        for year in np.unique(year_axis):
            selected = year_axis == year
            denominator = int(counts[selected].sum())
            numerator = int(available[selected].sum())
            by_year[str(year)] = {
                "member_cells": denominator,
                "available_cells": numerator,
                "missing_cells": denominator - numerator,
                "coverage": numerator / denominator if denominator else None,
            }
        denominator = int(counts.sum())
        entries[metadata.name] = {
            "metadata": metadata.as_dict(),
            "source": "sealed_local_runtime",
            "required_fields": list(metadata.required_fields),
            "member_cells": denominator,
            "missing_cells": denominator - int(available.sum()),
            "coverage": int(available.sum()) / denominator if denominator else None,
            "minimum_daily_coverage": float(ratio[counts > 0].min()) if np.any(counts > 0) else None,
            "first_available": str(dates[observed[0]]) if len(observed) else None,
            "last_available": str(dates[observed[-1]]) if len(observed) else None,
            "fully_missing_intervals": intervals,
            "by_year": by_year,
        }
    return {
        "schema_version": "wbr-factor-coverage-v1-pit-members",
        "factor_schema_hash": factors.schema_hash,
        "runtime_schema_hash": factors.runtime_schema_hash,
        "start": str(dates[0]),
        "end": str(dates[-1]),
        "denominator": "listing_age>=0 and not delisted; suspended members included",
        "factors": entries,
    }
