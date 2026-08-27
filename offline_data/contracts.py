"""Offline runtime contracts shared by factor, env, and training assembly."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np


RUNTIME_SCHEMA_VERSION = "wbr.runtime-slice.v1"
RUNTIME_LINEAGE_VERSION = "wbr.runtime-lineage.v1"
RUNTIME_GENERATION_SEMANTICS_VERSION = "wbr.runtime-generation-semantics.v1"

LEGACY_RUNTIME_PROVENANCE_NOTE = (
    "Legacy runtime NPZ files do not embed upstream source provenance; "
    "generation_semantics_sha256 verifies current builder/PIT code semantics "
    "only and cannot prove which upstream inputs or code produced the file."
)


@dataclass(frozen=True, slots=True)
class RuntimeFieldMetadata:
    """A registered runtime field and its decision-time availability."""

    name: str
    dimensions: tuple[str, ...]
    dtype: str
    decision_lag: int

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dimensions": list(self.dimensions),
            "dtype": self.dtype,
            "decision_lag": self.decision_lag,
        }


@dataclass(frozen=True, slots=True)
class RuntimeManifest:
    """Serializable identity and boundaries of one immutable runtime slice."""

    schema_version: str
    schema_hash: str
    source_path: str
    source_sha256: str
    source_size: int
    stock_vocabulary_sha256: str
    requested_start: str
    requested_end: str
    loaded_start: str
    loaded_end: str
    requested_preload_rows: int
    actual_preload_rows: int
    fields: tuple[RuntimeFieldMetadata, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "schema_hash": self.schema_hash,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "source_size": self.source_size,
            "stock_vocabulary_sha256": self.stock_vocabulary_sha256,
            "requested_start": self.requested_start,
            "requested_end": self.requested_end,
            "loaded_start": self.loaded_start,
            "loaded_end": self.loaded_end,
            "requested_preload_rows": self.requested_preload_rows,
            "actual_preload_rows": self.actual_preload_rows,
            "fields": [field.as_dict() for field in self.fields],
        }


@dataclass(frozen=True, slots=True)
class RuntimeFieldLineage:
    """Canonical identity of one registered field at a lineage cutoff."""

    name: str
    dimensions: tuple[str, ...]
    decision_lag: int
    source_dtype: str
    canonical_dtype: str
    canonical_shape: tuple[int, ...]
    semantic_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dimensions": list(self.dimensions),
            "decision_lag": self.decision_lag,
            "source_dtype": self.source_dtype,
            "canonical_dtype": self.canonical_dtype,
            "canonical_shape": list(self.canonical_shape),
            "semantic_sha256": self.semantic_sha256,
        }


@dataclass(frozen=True, slots=True)
class RuntimeGenerationComponent:
    """One source component contributing to runtime generation semantics."""

    name: str
    source_sha256: str

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True, slots=True)
class RuntimeLineage:
    """Versioned semantic identity for an append-only runtime prefix.

    Existing NPZ files contain no embedded upstream provenance manifest. The
    generation hash therefore verifies the current runtime/PIT code semantics,
    not the historical files, API responses, or exact code that built an old
    NPZ. ``upstream_provenance_embedded`` remains false for this contract.
    """

    lineage_version: str
    runtime_schema_version: str
    runtime_schema_hash: str
    generation_semantics_version: str
    generation_semantics_sha256: str
    generation_components: tuple[RuntimeGenerationComponent, ...]
    upstream_provenance_embedded: bool
    provenance_note: str
    semantic_sha256: str
    requested_cutoff: str | None
    prefix_start: str
    prefix_end: str
    prefix_rows: int
    stock_count: int
    stock_vocabulary_sha256: str
    date_axis_sha256: str
    fields: tuple[RuntimeFieldLineage, ...]

    @property
    def prefix_sha256(self) -> str:
        return self.semantic_sha256

    def field(self, name: str) -> RuntimeFieldLineage:
        for field in self.fields:
            if field.name == name:
                return field
        raise KeyError(name)

    def as_dict(self) -> dict[str, object]:
        return {
            "lineage_version": self.lineage_version,
            "runtime_schema_version": self.runtime_schema_version,
            "runtime_schema_hash": self.runtime_schema_hash,
            "generation_semantics_version": self.generation_semantics_version,
            "generation_semantics_sha256": self.generation_semantics_sha256,
            "generation_components": [
                component.as_dict() for component in self.generation_components
            ],
            "upstream_provenance_embedded": self.upstream_provenance_embedded,
            "provenance_note": self.provenance_note,
            "semantic_sha256": self.semantic_sha256,
            "requested_cutoff": self.requested_cutoff,
            "prefix_start": self.prefix_start,
            "prefix_end": self.prefix_end,
            "prefix_rows": self.prefix_rows,
            "stock_count": self.stock_count,
            "stock_vocabulary_sha256": self.stock_vocabulary_sha256,
            "date_axis_sha256": self.date_axis_sha256,
            "fields": [field.as_dict() for field in self.fields],
        }


@dataclass(frozen=True, slots=True)
class RuntimeSlice:
    """A copied, contiguous, read-only view of a sealed runtime date range.

    ``trade_dates`` and every array in ``data`` include preload rows. The
    decision interval is ``[decision_start, decision_stop)`` and the final
    loaded row is never later than the requested end date.
    """

    stock_codes: tuple[str, ...]
    trade_dates: np.ndarray
    data: Mapping[str, np.ndarray]
    decision_start: int
    decision_stop: int
    manifest: RuntimeManifest

    def __post_init__(self) -> None:
        dates = self.trade_dates
        if dates.ndim != 1 or dates.dtype != np.dtype("datetime64[D]"):
            raise ValueError("trade_dates must be a 1D datetime64[D] array")
        if not 0 <= self.decision_start < self.decision_stop <= len(dates):
            raise ValueError("invalid decision interval")
        if len(self.stock_codes) == 0:
            raise ValueError("stock vocabulary must not be empty")

        n_dates = len(dates)
        n_stocks = len(self.stock_codes)
        for name, values in self.data.items():
            if values.ndim == 2 and values.shape != (n_dates, n_stocks):
                raise ValueError(
                    f"runtime field {name!r} must have shape "
                    f"({n_dates}, {n_stocks})"
                )
            if values.ndim == 1 and values.shape != (n_stocks,):
                raise ValueError(
                    f"runtime field {name!r} must have shape ({n_stocks},)"
                )
            if values.ndim not in (1, 2):
                raise ValueError(f"runtime field {name!r} must be 1D or 2D")

        object.__setattr__(self, "data", MappingProxyType(dict(self.data)))

    @property
    def decision_dates(self) -> np.ndarray:
        return self.trade_dates[self.decision_start : self.decision_stop]

    @property
    def n_dates(self) -> int:
        return len(self.trade_dates)

    @property
    def n_stocks(self) -> int:
        return len(self.stock_codes)

    def field(self, name: str) -> np.ndarray:
        return self.data[name]

    def index_of(self, value: object) -> int:
        target = np.datetime64(value, "D")
        index = int(np.searchsorted(self.trade_dates, target))
        if index == len(self.trade_dates) or self.trade_dates[index] != target:
            raise KeyError(f"date is not present in runtime slice: {target}")
        return index
