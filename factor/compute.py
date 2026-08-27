"""One-time causal production-factor and soft-filter precomputation."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from offline_data import RuntimeSlice

from .base import FACTOR_SCHEMA_VERSION, FactorBatch, FactorDefinition
from .registry import PRODUCTION_FACTORS, PRODUCTION_FILTERS


def _lag_one(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    result[0] = np.nan
    result[1:] = values[:-1]
    return result


def _factor_panel(
    runtime: RuntimeSlice,
    definition: FactorDefinition,
    *,
    calculation_dtype: np.dtype | type | None = None,
) -> dict[str, np.ndarray]:
    panel = dict(runtime.data)
    if calculation_dtype is not None:
        for field in definition.metadata.required_fields:
            panel[field] = runtime.field(field).astype(
                calculation_dtype,
                copy=False,
            )
    for field in definition.metadata.lagged_fields:
        panel[field] = _lag_one(panel[field])
    return panel


def _validate_required_fields(
    runtime: RuntimeSlice,
    definitions: tuple[FactorDefinition, ...],
) -> None:
    required = {
        field
        for definition in definitions
        for field in definition.metadata.required_fields
    }
    missing = sorted(required.difference(runtime.data))
    if missing:
        raise ValueError(f"runtime slice is missing factor fields: {missing}")


def _cross_sectional_ranks(raw: np.ndarray) -> np.ndarray:
    """Daily ranks matching the existing compressed-valid-row semantics."""

    ranks = np.zeros(raw.shape, dtype=np.float32)
    for date_index, row in enumerate(raw):
        valid_mask = np.isfinite(row)
        valid_columns = np.flatnonzero(valid_mask)
        n_valid = valid_mask.sum()
        if n_valid == 0:
            continue
        order = np.argsort(row[valid_mask])[::-1]
        ranks[date_index, valid_columns[order]] = (
            1.0
            - np.arange(int(n_valid), dtype=np.float32) / n_valid
        )
    return ranks


def _schema_hash(
    factors: tuple[FactorDefinition, ...],
    filters: tuple[FactorDefinition, ...],
) -> str:
    payload = {
        "schema_version": FACTOR_SCHEMA_VERSION,
        "factors": [item.metadata.as_dict() for item in factors],
        "filters": [item.metadata.as_dict() for item in filters],
        "array_layout": "date,factor,stock",
        "calculation_dtype": "float64-from-runtime-cache-v2",
        "rank_semantics": (
            "float64-daily-descending-best-one-invalid-zero-v2"
        ),
        "raw_cache_semantics": "finite-clip-to-float32-range-v1",
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _freeze(values: np.ndarray) -> np.ndarray:
    values.flags.writeable = False
    return values


def _rank_universe(
    runtime: RuntimeSlice,
    rank_universe_mask: np.ndarray | None,
) -> tuple[np.ndarray, str]:
    if rank_universe_mask is None:
        mask = np.ones(runtime.n_stocks, dtype=np.bool_)
    else:
        mask = np.asarray(rank_universe_mask, dtype=np.bool_)
        if mask.shape != (runtime.n_stocks,):
            raise ValueError(
                "rank_universe_mask must match the fixed stock vocabulary"
            )
        mask = np.array(mask, dtype=np.bool_, order="C", copy=True)
    if not np.any(mask):
        raise ValueError("rank_universe_mask must contain at least one stock")
    digest = hashlib.sha256(mask.tobytes(order="C")).hexdigest()
    mask.flags.writeable = False
    return mask, digest


def precompute_factors(
    runtime: RuntimeSlice,
    rank_universe_mask: np.ndarray | None = None,
) -> FactorBatch:
    """Compute every registered production factor and filter exactly once.

    The API intentionally accepts no weights, so a zero static weight cannot
    suppress a factor needed by a later dynamic PPO action.
    """

    _validate_required_fields(runtime, PRODUCTION_FACTORS + PRODUCTION_FILTERS)
    universe_mask, universe_hash = _rank_universe(
        runtime,
        rank_universe_mask,
    )
    shape = (runtime.n_dates, len(PRODUCTION_FACTORS), runtime.n_stocks)
    raw_cache = np.empty(shape, dtype=np.float32)
    rank_cache = np.empty(shape, dtype=np.float32)
    validity_cache = np.empty(shape, dtype=np.bool_)
    float32_max = np.float64(np.finfo(np.float32).max)

    for index, definition in enumerate(PRODUCTION_FACTORS):
        with np.errstate(all="ignore"):
            calculated = np.ascontiguousarray(
                definition.implementation().calc_batch(
                    _factor_panel(
                        runtime,
                        definition,
                        calculation_dtype=np.float64,
                    )
                ),
                dtype=np.float64,
            )
        expected_shape = (runtime.n_dates, runtime.n_stocks)
        if calculated.shape != expected_shape:
            raise ValueError(
                f"factor {definition.metadata.name} returned "
                f"{calculated.shape}, expected {expected_shape}"
            )
        calculated_finite = np.isfinite(calculated)
        np.clip(
            calculated,
            -float32_max,
            float32_max,
            out=raw_cache[:, index, :],
        )
        raw_cache[:, index, :][~calculated_finite] = calculated[
            ~calculated_finite
        ]
        validity_cache[:, index, :] = (
            calculated_finite & universe_mask[None, :]
        )
        rank_cache[:, index, :] = 0.0
        rank_cache[:, index, :][:, universe_mask] = _cross_sectional_ranks(
            calculated[:, universe_mask]
        )

    filter_shape = (
        runtime.n_dates,
        len(PRODUCTION_FILTERS),
        runtime.n_stocks,
    )
    filter_cache = np.empty(filter_shape, dtype=np.bool_)
    for index, definition in enumerate(PRODUCTION_FILTERS):
        calculated = np.asarray(
            definition.implementation().calc_batch(
                _factor_panel(runtime, definition)
            )
        )
        expected_shape = (runtime.n_dates, runtime.n_stocks)
        if calculated.shape != expected_shape:
            raise ValueError(
                f"filter {definition.metadata.name} returned "
                f"{calculated.shape}, expected {expected_shape}"
            )
        filter_cache[:, index, :] = np.isfinite(calculated) & (calculated > 0)

    return FactorBatch(
        schema_version=FACTOR_SCHEMA_VERSION,
        schema_hash=_schema_hash(PRODUCTION_FACTORS, PRODUCTION_FILTERS),
        runtime_schema_hash=runtime.manifest.schema_hash,
        rank_universe_sha256=universe_hash,
        stock_codes=runtime.stock_codes,
        trade_dates=runtime.trade_dates,
        decision_start=runtime.decision_start,
        decision_stop=runtime.decision_stop,
        factor_metadata=tuple(item.metadata for item in PRODUCTION_FACTORS),
        filter_metadata=tuple(item.metadata for item in PRODUCTION_FILTERS),
        raw=_freeze(raw_cache),
        ranks=_freeze(rank_cache),
        validity=_freeze(validity_cache),
        filters=_freeze(filter_cache),
        rank_universe_mask=universe_mask,
    )
