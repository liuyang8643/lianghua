"""Public factor metadata and precomputed batch contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


FACTOR_SCHEMA_VERSION = "wbr.production-factors.v10-selected12-completed-amihud"


@dataclass(frozen=True, slots=True)
class FactorMetadata:
    name: str
    version: str
    hist_days: int
    required_fields: tuple[str, ...]
    implementation_hash: str
    lagged_fields: tuple[str, ...] = ()
    score_semantics: Literal["continuous", "binary"] = "continuous"

    def __post_init__(self) -> None:
        if self.score_semantics not in ("continuous", "binary"):
            raise ValueError("score_semantics must be continuous or binary")

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "hist_days": self.hist_days,
            "required_fields": list(self.required_fields),
            "implementation_hash": self.implementation_hash,
            "lagged_fields": list(self.lagged_fields),
            "score_semantics": self.score_semantics,
        }


@dataclass(frozen=True, slots=True)
class FactorDefinition:
    """One implementation and its declared input representation.

    The default preserves canonical float64 calculation inputs. Explicit
    ``raw_runtime_view`` definitions instead receive the original read-only
    arrays and own any numeric conversion; implicit lag transforms are then
    forbidden so object identity remains a meaningful binding contract.
    """

    metadata: FactorMetadata
    implementation: type[Any]
    raw_runtime_view: bool = False

    def __post_init__(self) -> None:
        if type(self.raw_runtime_view) is not bool:
            raise TypeError("raw_runtime_view must be bool")
        if self.raw_runtime_view and self.metadata.lagged_fields:
            raise ValueError("raw runtime views cannot request transformed lagged fields")


@dataclass(frozen=True, slots=True)
class FactorDay:
    index: int
    trade_date: np.datetime64
    factor_names: tuple[str, ...]
    filter_names: tuple[str, ...]
    raw: np.ndarray
    ranks: np.ndarray
    validity: np.ndarray
    filters: np.ndarray


@dataclass(frozen=True, slots=True)
class FactorBatch:
    """One-time factor cache aligned exactly to a RuntimeSlice.

    Factor arrays use ``[date, factor, stock]``. Filter arrays use
    ``[date, filter, stock]``. Invalid ranks are represented by zero and must
    always be interpreted together with ``validity``. Nonmembers also receive
    rank zero; raw validity describes input availability, not pool membership.
    """

    schema_version: str
    schema_hash: str
    runtime_schema_hash: str
    stock_codes: tuple[str, ...]
    trade_dates: np.ndarray
    decision_start: int
    decision_stop: int
    factor_metadata: tuple[FactorMetadata, ...]
    filter_metadata: tuple[FactorMetadata, ...]
    raw: np.ndarray
    ranks: np.ndarray
    validity: np.ndarray
    filters: np.ndarray

    def __post_init__(self) -> None:
        n_dates = len(self.trade_dates)
        n_stocks = len(self.stock_codes)
        n_factors = len(self.factor_metadata)
        n_filters = len(self.filter_metadata)
        factor_shape = (n_dates, n_factors, n_stocks)
        filter_shape = (n_dates, n_filters, n_stocks)
        if self.raw.shape != factor_shape:
            raise ValueError(f"raw must have shape {factor_shape}")
        if self.ranks.shape != factor_shape:
            raise ValueError(f"ranks must have shape {factor_shape}")
        if self.validity.shape != factor_shape:
            raise ValueError(f"validity must have shape {factor_shape}")
        if self.filters.shape != filter_shape:
            raise ValueError(f"filters must have shape {filter_shape}")
        if not 0 <= self.decision_start < self.decision_stop <= n_dates:
            raise ValueError("invalid factor decision interval")

    @property
    def factor_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.factor_metadata)

    @property
    def filter_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.filter_metadata)

    @property
    def decision_dates(self) -> np.ndarray:
        return self.trade_dates[self.decision_start : self.decision_stop]

    def index_of(self, value: object) -> int:
        target = np.datetime64(value, "D")
        index = int(np.searchsorted(self.trade_dates, target))
        if index == len(self.trade_dates) or self.trade_dates[index] != target:
            raise KeyError(f"date is not present in factor batch: {target}")
        return index

    def day(self, index: int) -> FactorDay:
        if not 0 <= index < len(self.trade_dates):
            raise IndexError(index)
        return FactorDay(
            index=index,
            trade_date=self.trade_dates[index],
            factor_names=self.factor_names,
            filter_names=self.filter_names,
            raw=self.raw[index],
            ranks=self.ranks[index],
            validity=self.validity[index],
            filters=self.filters[index],
        )
