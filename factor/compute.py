"""One-time causal production-factor and soft-filter precomputation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np
from numba import njit

from offline_data import RuntimeSlice
from utils.stable_sort import stable_radix_order

from .base import FACTOR_SCHEMA_VERSION, FactorBatch, FactorDefinition
from .registry import PRODUCTION_FACTORS, PRODUCTION_FILTERS
from .library.bilibili import (
    BiliAdjustedIssueDiscount, BiliHighLifetimeRangeRatio, calculate_bilibili_scores,
)
from .library.styles import CompletedReversal20, CompletedMomentum252Skip21, _matrices, _official_log_returns


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
    panel["trade_dates"] = runtime.trade_dates
    if definition.raw_runtime_view:
        if any(panel[field].flags.writeable for field in definition.metadata.required_fields):
            raise ValueError("raw runtime factor inputs must be read-only")
    elif calculation_dtype is not None:
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


@njit(cache=True, fastmath=False, parallel=False)
def _score_cache_kernel(
    values, membership, raw, validity, ranks, binary, key_kind,
):
    """One IEEE rank/cache writer; all workspaces are only stock-sized.

    Float keys retain every float64 distinction; signed and unsigned integer
    keys retain distinctions above 2**53. Equal scores share average descending
    positions, including +/-zero. Raw validity does not imply PIT membership.
    """
    rows, stocks = values.shape
    scores = np.empty(stocks, dtype=np.float64)
    bits = scores.view(np.uint64)
    keys = np.empty(stocks, dtype=np.uint64)
    columns = np.empty(stocks, dtype=np.int64)
    temporary_columns = np.empty(stocks, dtype=np.int64)
    buckets = np.empty(256, dtype=np.int64)
    sign_bit = np.uint64(1) << np.uint64(63)
    maximum = np.float64(np.finfo(np.float32).max)
    for row in range(rows):
        count = 0
        for column in range(stocks):
            value = values[row, column]
            finite = np.isfinite(value)
            if raw is not None:
                validity[row, column] = finite
                raw_value = value
                if finite:
                    raw_value = min(max(raw_value, -maximum), maximum)
                raw[row, column] = raw_value
            ranks[row, column] = np.float32(0.0)
            if finite and (membership is None or membership[row, column]):
                if binary:
                    if value != 0.0 and value != 1.0:
                        raise ValueError("binary factor scores must be 0 or 1 when finite")
                    ranks[row, column] = value
                else:
                    if key_kind == 1:
                        keys[column] = np.uint64(value) ^ sign_bit
                    elif key_kind == 2:
                        keys[column] = np.uint64(value)
                    else:
                        scores[count] = value
                        bit_value = bits[count]
                        keys[column] = ~bit_value if (bit_value & sign_bit) else (bit_value ^ sign_bit)
                    columns[count] = column
                    count += 1
        if count == 0:
            continue
        columns, temporary_columns = stable_radix_order(
            keys, columns, temporary_columns, count, buckets,
        )
        denominator = np.float32(count)
        start = 0
        while start < count:
            stop = start + 1
            value = values[row, columns[start]]
            while stop < count and values[row, columns[stop]] == value:
                stop += 1
            position = np.float32((2 * count - start - stop - 1) * 0.5)
            rank = np.float32(1.0) - position / denominator
            for index in range(start, stop):
                ranks[row, columns[index]] = rank
            start = stop


def scores_to_ranks(
    scores: np.ndarray,
    *,
    score_semantics: str = "continuous",
    membership: np.ndarray | None = None,
) -> np.ndarray:
    """Average descending positional ranks, or preserve declared binary 0/1.

    Continuous ties share their average position, using the same denominator
    as untied scores. Invalid values receive zero and require a validity mask.
    Binary values never depend on group size. The denominator is the number
    of valid members on that date, never the final runtime stock-axis size.
    """

    values = np.asarray(scores)
    if values.ndim != 2:
        raise ValueError("scores must be a two-dimensional date-by-stock matrix")
    if score_semantics not in ("continuous", "binary"):
        raise ValueError("score_semantics must be continuous or binary")
    if membership is not None and (membership.shape != values.shape or membership.dtype != np.bool_):
        raise ValueError("rank membership must be a matching boolean date-by-stock matrix")
    # The compiled numeric kernel consumes native byte order. Widen float16
    # losslessly; keep float32/64 and integer precision without unnecessary copies.
    input_dtype = values.dtype.newbyteorder("=")
    if input_dtype.kind == "f":
        input_dtype = np.promote_types(input_dtype, np.dtype(np.float32))
    values = values.astype(input_dtype, copy=False)
    ranks = np.empty(values.shape, dtype=np.float32)
    key_kind = 1 if values.dtype.kind == "i" else (2 if values.dtype.kind in "ub" else 0)
    _score_cache_kernel(
        values, membership, None, None,
        ranks, score_semantics == "binary", key_kind,
    )
    return ranks


def _schema_hash(
    factors: tuple[FactorDefinition, ...],
    filters: tuple[FactorDefinition, ...],
    schema_version: str = FACTOR_SCHEMA_VERSION,
) -> str:
    payload = {
        "schema_version": schema_version,
        "factors": [item.metadata.as_dict() for item in factors],
        "filters": [item.metadata.as_dict() for item in filters],
        "array_layout": "date,factor,stock",
        "calculation_dtype": "float64-from-runtime-cache-v2",
        "rank_semantics": (
            "float64-pit-member-average-positional-ties-binary-preserve-01-"
            "nonmember-invalid-zero-v5"
        ),
        "raw_cache_semantics": "finite-clip-to-float32-range-v1",
    }
    raw_view_names = [
        item.metadata.name for item in (*factors, *filters) if item.raw_runtime_view
    ]
    if raw_view_names:
        payload["raw_runtime_view_inputs"] = raw_view_names
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


def precompute_factors(
    runtime: RuntimeSlice,
    *,
    definitions: tuple[FactorDefinition, ...] = PRODUCTION_FACTORS,
) -> FactorBatch:
    """Compute an explicit vocabulary and the production filters once.

    The API intentionally accepts no weights, so a zero static weight cannot
    suppress a factor needed by a later dynamic PPO action. Research callers
    supply versioned definitions; their batch has a separate schema identity
    and does not modify the production registry or actor vocabulary.
    """

    if runtime.manifest.replay_projection is not None:
        raise ValueError("replay projection contains precomputed factors; reload the original runtime to recompute")
    if not isinstance(definitions, tuple) or not definitions or any(
        not isinstance(item, FactorDefinition) for item in definitions
    ):
        raise TypeError("definitions must be a non-empty tuple of FactorDefinition")
    names = tuple(item.metadata.name for item in definitions)
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("factor definition names must be non-empty and unique")
    schema_version = (
        FACTOR_SCHEMA_VERSION if definitions == PRODUCTION_FACTORS
        else "wbr.research-factors.v1-pit-ranks"
    )
    _validate_required_fields(runtime, definitions + PRODUCTION_FILTERS)
    shape = (runtime.n_dates, len(definitions), runtime.n_stocks)
    raw_cache = np.empty(shape, dtype=np.float32)
    rank_cache = np.empty(shape, dtype=np.float32)
    validity_cache = np.empty(shape, dtype=np.bool_)
    member = (runtime.field("listing_age") >= 0) & ~runtime.field("delisted_mask")

    lifetime_classes = (BiliAdjustedIssueDiscount, BiliHighLifetimeRangeRatio)
    lifetime_names = tuple(
        item.metadata.name for item in definitions
        if item.implementation in lifetime_classes
    )
    lifetime_scores = {}
    if lifetime_names:
        if runtime.manifest.actual_preload_rows >= runtime.manifest.requested_preload_rows:
            raise ValueError("lifetime factors require preload extending to the runtime's first row")
        lifetime_scores = dict(calculate_bilibili_scores(
            runtime.trade_dates, runtime.data, factor_names=lifetime_names,
        ))

    completed_classes = (CompletedReversal20, CompletedMomentum252Skip21)
    def completed_input_key(definition):
        return (definition.raw_runtime_view,
                tuple(definition.metadata.lagged_fields.count(name) for name in ("close", "preClose")))
    completed_counts = Counter(completed_input_key(item) for item in definitions
                               if item.implementation in completed_classes)
    completed_returns = {}
    for index, definition in enumerate(definitions):
        implementation = definition.implementation
        if implementation in completed_classes:
            completed_key = completed_input_key(definition)
            if completed_key not in completed_returns:
                panel = _factor_panel(runtime, definition, calculation_dtype=np.float64)
                close, pre_close = _matrices(panel, "close", "preClose")
                completed_returns[completed_key] = _official_log_returns(close, pre_close)
                del panel, close, pre_close
        with np.errstate(all="ignore"):
            if implementation in lifetime_classes:
                calculated = lifetime_scores.pop(definition.metadata.name)
            elif implementation in completed_classes:
                calculated = implementation().calc_from_returns(completed_returns[completed_key])
            else:
                calculated = implementation().calc_batch(
                    _factor_panel(runtime, definition, calculation_dtype=np.float64)
                )
            calculated = np.ascontiguousarray(calculated, dtype=np.float64)
        if implementation in completed_classes:
            completed_counts[completed_key] -= 1
            if completed_counts[completed_key] == 0:
                del completed_returns[completed_key]
        expected_shape = (runtime.n_dates, runtime.n_stocks)
        if calculated.shape != expected_shape:
            raise ValueError(
                f"factor {definition.metadata.name} returned "
                f"{calculated.shape}, expected {expected_shape}"
            )
        _score_cache_kernel(
            calculated, member, raw_cache[:, index, :],
            validity_cache[:, index, :], rank_cache[:, index, :],
            definition.metadata.score_semantics == "binary", 0,
        )
        # Do not retain the previous full-history result while allocating the
        # next factor's inputs. The immutable output cache now owns its values.
        del calculated

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
        del calculated

    return FactorBatch(
        schema_version=schema_version,
        schema_hash=_schema_hash(definitions, PRODUCTION_FILTERS, schema_version),
        runtime_schema_hash=runtime.manifest.schema_hash,
        stock_codes=runtime.stock_codes,
        trade_dates=runtime.trade_dates,
        decision_start=runtime.decision_start,
        decision_stop=runtime.decision_stop,
        factor_metadata=tuple(item.metadata for item in definitions),
        filter_metadata=tuple(item.metadata for item in PRODUCTION_FILTERS),
        raw=_freeze(raw_cache),
        ranks=_freeze(rank_cache),
        validity=_freeze(validity_cache),
        filters=_freeze(filter_cache),
    )
