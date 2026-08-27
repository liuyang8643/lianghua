"""Explicit production factor vocabulary; no candidate-directory scanning."""

from __future__ import annotations

import hashlib
import inspect
import json
from types import ModuleType

from factor_db.factors.AmihudIlliquidity import AmihudIlliquidity
from factor_db.factors.AmountBasedSmallCap import AmountBasedSmallCap
from factor_db.factors.TrueMarketCap import TrueMarketCap
from factor_db.factors.VolumeCV import VolumeCV
from factor_db.factors.filter import FilterLowPrice, FilterST, FilterStarST

from .base import FactorDefinition, FactorMetadata


PRODUCTION_FACTOR_NAMES = (
    "AmihudIlliquidity",
    "TrueMarketCap",
    "VolumeCV",
    "AmountBasedSmallCap",
)
PRODUCTION_FILTER_NAMES = ("FilterST", "FilterStarST", "FilterLowPrice")


def _implementation_hash(
    implementation: type,
    *,
    version: str,
    lagged_fields: tuple[str, ...] = (),
) -> str:
    module = inspect.getmodule(implementation)
    if not isinstance(module, ModuleType):
        raise TypeError(f"cannot resolve module for {implementation.__name__}")
    payload = {
        "name": implementation.__name__,
        "version": version,
        "lagged_fields": lagged_fields,
        "source": inspect.getsource(module),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _definition(
    implementation: type,
    *,
    version: str,
    required_fields: tuple[str, ...],
    lagged_fields: tuple[str, ...] = (),
    hist_days: int | None = None,
) -> FactorDefinition:
    effective_hist_days = (
        int(implementation.hist_days) if hist_days is None else int(hist_days)
    )
    metadata = FactorMetadata(
        name=implementation.__name__,
        version=version,
        hist_days=effective_hist_days,
        required_fields=required_fields,
        implementation_hash=_implementation_hash(
            implementation,
            version=version,
            lagged_fields=lagged_fields,
        ),
        lagged_fields=lagged_fields,
    )
    return FactorDefinition(metadata=metadata, implementation=implementation)


PRODUCTION_FACTORS: tuple[FactorDefinition, ...] = (
    _definition(
        AmihudIlliquidity,
        version="legacy-unadjusted-return-v1",
        required_fields=("close", "amount"),
    ),
    _definition(
        TrueMarketCap,
        version="t-open-lagged-total-share-v1",
        required_fields=("open", "total_share"),
        lagged_fields=("total_share",),
        hist_days=1,
    ),
    _definition(
        VolumeCV,
        version="legacy-known-volume-v1",
        required_fields=("volume",),
    ),
    _definition(
        AmountBasedSmallCap,
        version="legacy-known-amount-v1",
        required_fields=("amount",),
    ),
)

PRODUCTION_FILTERS: tuple[FactorDefinition, ...] = (
    _definition(
        FilterST,
        version="runtime-st-mask-v1",
        required_fields=("st_mask",),
    ),
    _definition(
        FilterStarST,
        version="star-mask-or-conservative-st-fallback-v1",
        required_fields=("st_mask",),
    ),
    _definition(
        FilterLowPrice,
        version="t-open-min-price-2-v1",
        required_fields=("open",),
    ),
)

if tuple(item.metadata.name for item in PRODUCTION_FACTORS) != PRODUCTION_FACTOR_NAMES:
    raise RuntimeError("production factor vocabulary order changed")
if tuple(item.metadata.name for item in PRODUCTION_FILTERS) != PRODUCTION_FILTER_NAMES:
    raise RuntimeError("production filter vocabulary order changed")


def get_factor_definition(name: str) -> FactorDefinition:
    return {item.metadata.name: item for item in PRODUCTION_FACTORS}[name]


def get_filter_definition(name: str) -> FactorDefinition:
    return {item.metadata.name: item for item in PRODUCTION_FILTERS}[name]
