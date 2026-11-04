"""Opt-in optimized factor catalog; does not change any strategy defaults."""
from __future__ import annotations

import hashlib
import inspect

from factor.base import FactorDefinition, FactorMetadata
from factor.library import bilibili, completed_windows, optimized_eight as signals


def _definition(name, fields, *, lagged=(), binary=False, history=None):
    implementation = getattr(signals, name)
    # Bind inherited implementations and shared rolling/IPO kernels as well.
    modules = {signals, completed_windows, bilibili}
    modules.update(inspect.getmodule(cls) for cls in implementation.__mro__ if cls is not object)
    source = "\n".join(inspect.getsource(module) for module in sorted(modules, key=lambda m: m.__name__))
    return FactorDefinition(FactorMetadata(
        name=name,
        version="optimized-individual-factors-20261104-v1",
        hist_days=implementation.hist_days if history is None else history,
        required_fields=fields,
        lagged_fields=lagged,
        score_semantics="binary" if binary else "continuous",
        implementation_hash=hashlib.sha256(source.encode("utf-8")).hexdigest(),
    ), implementation)


# This is a catalog of individual definitions, not a portfolio/weight preset.
OPTIMIZED_FACTORS = (
    _definition("TrueMarketCap", ("open", "total_share", "st_mask"), lagged=("total_share",), history=1),
    _definition("VolumeCV", ("volume", "open", "st_mask")),
    _definition("AmountBasedSmallCap5", ("amount", "open", "st_mask")),
    _definition("BiliAdjustedIssueDiscount", bilibili.REQUIRED_FIELDS),
    _definition("PB3ROE5Signal", ("open", "total_share", "financial_profit_ttm", "financial_equity"), lagged=("total_share",), binary=True),
    _definition("ProfitGrowthOriginalAvailability", ("financial_profit_yoy", "financial_cash_outflow_yoy")),
    _definition("BiliLowReturnAttention10", ("close", "preClose", "listing_age", "delisted_mask")),
    _definition("LowResidualVol20", ("close", "preClose", "listing_age", "delisted_mask")),
)
OPTIMIZED_FACTOR_NAMES = tuple(item.metadata.name for item in OPTIMIZED_FACTORS)
_BY_NAME = {item.metadata.name: item for item in OPTIMIZED_FACTORS}


def get_optimized_factor_definition(name: str) -> FactorDefinition:
    """Return one factor for explicit use with precompute_factors(definitions=...)."""
    return _BY_NAME[name]
