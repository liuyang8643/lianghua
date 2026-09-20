"""Explicit production factor vocabulary; no candidate-directory scanning."""

from __future__ import annotations

import hashlib
import inspect
import json
from types import ModuleType

from factor_db.factors.AmountBasedSmallCap import AmountBasedSmallCap
from factor_db.factors.TrueMarketCap import TrueMarketCap
from factor_db.factors.VolumeCV import VolumeCV
from factor_db.factors.filter import FilterLowPrice, FilterST
from factor.library import (
    CompletedReversal20, CompletedMomentum252Skip21,
)
from factor.library.bilibili import (
    BiliAdjustedIssueDiscount, BiliHighLifetimeRangeRatio,
    REQUIRED_FIELDS as BILIBILI_REQUIRED_FIELDS,
    SEMANTICS_VERSION as BILIBILI_SEMANTICS_VERSION,
)
from factor.library.financial import (
    FINANCIAL_SEMANTICS_VERSION,
    PBBelowTwoROEAbove10Signal,
    LowCashOutflowProfitGrowthSpread,
    HighOperatingProfitRevenueGrowthSpread,
)
from factor.library.abnormal_gross_profit import (
    ABNORMAL_GROSS_PROFIT_VERSION,
    ABNORMAL_GROSS_PROFIT_PANEL_FIELDS,
    HighAbnormalGrossProfit,
)

from .base import FactorDefinition, FactorMetadata


PRODUCTION_FACTOR_NAMES = (
    "TrueMarketCap",
    "VolumeCV",
    "AmountBasedSmallCap",
    "CompletedReversal20",
    "CompletedMomentum252Skip21",
    "BiliAdjustedIssueDiscount",
    "BiliHighLifetimeRangeRatio",
    "PBBelowTwoROEAbove10Signal",
    "LowCashOutflowProfitGrowthSpread",
    "HighOperatingProfitRevenueGrowthSpread",
    "HighAbnormalGrossProfit",
)
PRODUCTION_FILTER_NAMES = ("FilterST", "FilterLowPrice")


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
    score_semantics: str = "continuous",
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
        score_semantics=score_semantics,
    )
    return FactorDefinition(metadata=metadata, implementation=implementation)


PRODUCTION_FACTORS: tuple[FactorDefinition, ...] = (
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
    _definition(CompletedReversal20, version="completed-official-return-v1", required_fields=("close", "preClose")),
    _definition(CompletedMomentum252Skip21, version="completed-official-return-v2-interior-skip-min185-endpoints", required_fields=("close", "preClose")),
    _definition(BiliAdjustedIssueDiscount, version=BILIBILI_SEMANTICS_VERSION, required_fields=BILIBILI_REQUIRED_FIELDS),
    _definition(BiliHighLifetimeRangeRatio, version=BILIBILI_SEMANTICS_VERSION, required_fields=BILIBILI_REQUIRED_FIELDS),
    _definition(
        PBBelowTwoROEAbove10Signal,
        version=FINANCIAL_SEMANTICS_VERSION,
        required_fields=("open", "total_share", "financial_profit_ttm", "financial_equity"),
        lagged_fields=("total_share",),
        score_semantics="binary",
    ),
    _definition(
        LowCashOutflowProfitGrowthSpread,
        version=FINANCIAL_SEMANTICS_VERSION,
        required_fields=("financial_cash_outflow_yoy", "financial_profit_yoy"),
    ),
    _definition(
        HighOperatingProfitRevenueGrowthSpread,
        version=FINANCIAL_SEMANTICS_VERSION,
        required_fields=("financial_operating_profit_yoy", "financial_revenue_yoy"),
    ),
    _definition(
        HighAbnormalGrossProfit,
        version=ABNORMAL_GROSS_PROFIT_VERSION,
        required_fields=ABNORMAL_GROSS_PROFIT_PANEL_FIELDS,
    ),
)

PRODUCTION_FILTERS: tuple[FactorDefinition, ...] = (
    _definition(
        FilterST,
        version="runtime-st-mask-v1",
        required_fields=("st_mask",),
    ),
    _definition(
        FilterLowPrice,
        version="t-open-min-price-2-v1",
        required_fields=("open",),
    ),
)

_PRODUCTION_DEFINITIONS = {
    item.metadata.name: item
    for item in (*PRODUCTION_FACTORS, *PRODUCTION_FILTERS)
}

if tuple(item.metadata.name for item in PRODUCTION_FACTORS) != PRODUCTION_FACTOR_NAMES:
    raise RuntimeError("production factor vocabulary order changed")
if tuple(item.metadata.name for item in PRODUCTION_FILTERS) != PRODUCTION_FILTER_NAMES:
    raise RuntimeError("production filter vocabulary order changed")


def get_factor_definition(name: str) -> FactorDefinition:
    definition = _PRODUCTION_DEFINITIONS[name]
    if name not in PRODUCTION_FACTOR_NAMES:
        raise KeyError(name)
    return definition


def get_filter_definition(name: str) -> FactorDefinition:
    definition = _PRODUCTION_DEFINITIONS[name]
    if name not in PRODUCTION_FILTER_NAMES:
        raise KeyError(name)
    return definition


def get_factor_class(name: str) -> type:
    """Return one explicitly registered production factor/filter class."""
    return _PRODUCTION_DEFINITIONS[name].implementation
