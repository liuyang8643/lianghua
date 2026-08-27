"""Strictly sealed, offline-only loading of local runtime NPZ snapshots."""

from __future__ import annotations

import hashlib
import json
import struct
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

from .contracts import (
    LEGACY_RUNTIME_PROVENANCE_NOTE,
    RUNTIME_GENERATION_SEMANTICS_VERSION,
    RUNTIME_LINEAGE_VERSION,
    RUNTIME_SCHEMA_VERSION,
    RuntimeFieldMetadata,
    RuntimeFieldLineage,
    RuntimeGenerationComponent,
    RuntimeLineage,
    RuntimeManifest,
    RuntimeSlice,
)


DEFAULT_LOOKBACK = 64
MAX_PRODUCTION_FACTOR_HISTORY = 60
MIN_PRELOAD_ROWS = 126

RUNTIME_FIELDS: tuple[RuntimeFieldMetadata, ...] = (
    RuntimeFieldMetadata("open", ("date", "stock"), "float32", 0),
    RuntimeFieldMetadata("high", ("date", "stock"), "float32", 1),
    RuntimeFieldMetadata("low", ("date", "stock"), "float32", 1),
    RuntimeFieldMetadata("close", ("date", "stock"), "float32", 1),
    RuntimeFieldMetadata("volume", ("date", "stock"), "float32", 1),
    RuntimeFieldMetadata("amount", ("date", "stock"), "float32", 1),
    RuntimeFieldMetadata("preClose", ("date", "stock"), "float32", 0),
    # Announcement timestamps are absent in the current runtime build.
    RuntimeFieldMetadata("total_share", ("date", "stock"), "float32", 1),
    RuntimeFieldMetadata("bps", ("date", "stock"), "float32", 0),
    RuntimeFieldMetadata("eps", ("date", "stock"), "float32", 0),
    RuntimeFieldMetadata("roe", ("date", "stock"), "float32", 0),
    RuntimeFieldMetadata("profit_yoy", ("date", "stock"), "float32", 0),
    RuntimeFieldMetadata("revenue_yoy", ("date", "stock"), "float32", 0),
    RuntimeFieldMetadata("operating_cf_ps", ("date", "stock"), "float32", 0),
    RuntimeFieldMetadata("gross_margin", ("date", "stock"), "float32", 0),
    RuntimeFieldMetadata("st_mask", ("date", "stock"), "bool", 0),
    RuntimeFieldMetadata("issue_price", ("stock",), "float32", 0),
    RuntimeFieldMetadata("stock_names", ("stock",), "str", 0),
)

OPTIONAL_RUNTIME_FIELDS: tuple[RuntimeFieldMetadata, ...] = (
    RuntimeFieldMetadata("star_st_mask", ("date", "stock"), "bool", 0),
)

_LINEAGE_CHUNK_ROWS = 64

_GENERATION_SEMANTICS_COMPONENTS = (
    ("runtime_builder", "data/build_runtime.py"),
    ("financial_pit", "data/financial_pit.py"),
    ("kline_and_preclose_builder", "data/kline_mootdx.py"),
    ("financial_source_builder", "data/update_financial_deep.py"),
    ("offline_source_orchestration", "data/update_all.py"),
    ("delist_semantics", "data/db/delist.py"),
    ("stock_name_semantics", "data/db/stock_name.py"),
    ("stock_classification", "utils/stock/info.py"),
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _schema_hash(fields: tuple[RuntimeFieldMetadata, ...]) -> str:
    payload = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "fields": [field.as_dict() for field in fields],
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _stock_vocabulary_hash(stock_codes: tuple[str, ...]) -> str:
    return hashlib.sha256(_canonical_json(list(stock_codes))).hexdigest()


def _update_framed(digest, payload: bytes) -> None:
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


@lru_cache(maxsize=8)
def _file_sha256_cached(path_text: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    source = Path(path).resolve()
    stat = source.stat()
    return _file_sha256_cached(str(source), stat.st_size, stat.st_mtime_ns)


def _as_date(value: object) -> np.datetime64:
    return np.datetime64(value, "D")


def _owned_read_only(
    values: np.ndarray,
    *,
    dtype: np.dtype | str | None = None,
) -> np.ndarray:
    result = np.array(values, dtype=dtype, order="C", copy=True)
    result.flags.writeable = False
    return result


def _validate_dates(values: np.ndarray) -> np.ndarray:
    dates = _owned_read_only(values, dtype="datetime64[D]")
    if dates.ndim != 1 or len(dates) == 0:
        raise ValueError("trade_dates must be a non-empty 1D array")
    if np.isnat(dates).any():
        raise ValueError("trade_dates must not contain NaT")
    if np.any(dates[1:] <= dates[:-1]):
        raise ValueError("trade_dates must be unique and strictly increasing")
    return dates


def _validate_stock_codes(values: np.ndarray) -> tuple[str, ...]:
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("stock_codes must be a non-empty 1D array")
    codes = tuple(
        unicodedata.normalize("NFC", code)
        for code in np.asarray(values).astype(str).tolist()
    )
    if any(not code for code in codes):
        raise ValueError("stock_codes must not contain empty values")
    if len(set(codes)) != len(codes):
        raise ValueError("stock_codes must be unique")
    return codes


def _selected_fields(
    npz_files: Iterable[str],
) -> tuple[RuntimeFieldMetadata, ...]:
    available = set(npz_files)
    missing = [field.name for field in RUNTIME_FIELDS if field.name not in available]
    if missing:
        raise ValueError(f"runtime NPZ is missing registered fields: {missing}")
    optional = tuple(
        field for field in OPTIONAL_RUNTIME_FIELDS if field.name in available
    )
    registered = {
        "stock_codes",
        "trade_dates",
        *(field.name for field in RUNTIME_FIELDS),
        *(field.name for field in OPTIONAL_RUNTIME_FIELDS),
    }
    unknown = sorted(available - registered)
    if unknown:
        raise ValueError(
            f"runtime NPZ has unregistered fields; upgrade the runtime and "
            f"observation schemas first: {unknown}"
        )
    return RUNTIME_FIELDS + optional


def _copy_field(
    source: np.ndarray,
    metadata: RuntimeFieldMetadata,
    row_start: int,
    row_stop: int,
    n_dates: int,
    n_stocks: int,
) -> np.ndarray:
    expected = _expected_field_shape(metadata, n_dates, n_stocks)
    if source.shape != expected:
        raise ValueError(
            f"runtime field {metadata.name!r} has shape {source.shape}, "
            f"expected {expected}"
        )
    selected = source[row_start:row_stop] if source.ndim == 2 else source
    dtype = str if metadata.dtype == "str" else metadata.dtype
    return _owned_read_only(selected, dtype=dtype)


def _expected_field_shape(
    metadata: RuntimeFieldMetadata,
    n_dates: int,
    n_stocks: int,
) -> tuple[int, ...]:
    return (
        (n_dates, n_stocks)
        if metadata.dimensions == ("date", "stock")
        else (n_stocks,)
    )


def _canonical_numeric_chunk(
    values: np.ndarray,
    dtype: str,
) -> np.ndarray:
    if dtype == "bool":
        return np.array(values, dtype=np.uint8, order="C", copy=True)
    if dtype != "float32":
        raise ValueError(f"unsupported canonical numeric dtype: {dtype}")
    with np.errstate(over="ignore", invalid="ignore"):
        normalized = np.array(values, dtype="<f4", order="C", copy=True)
    normalized[normalized == 0] = 0.0
    normalized[np.isnan(normalized)] = np.float32(np.nan)
    return normalized


def _hash_text_values(
    digest,
    values: np.ndarray,
) -> None:
    normalized = np.asarray(values).astype(str).reshape(-1)
    for value in normalized:
        encoded = unicodedata.normalize("NFC", str(value)).encode("utf-8")
        _update_framed(digest, encoded)


def _compute_field_lineage(
    metadata: RuntimeFieldMetadata,
    values: np.ndarray,
    *,
    prefix_rows: int,
    n_dates: int,
    n_stocks: int,
) -> RuntimeFieldLineage:
    expected = _expected_field_shape(metadata, n_dates, n_stocks)
    if values.shape != expected:
        raise ValueError(
            f"runtime field {metadata.name!r} has shape {values.shape}, "
            f"expected {expected}"
        )
    canonical_shape = (
        (prefix_rows, n_stocks)
        if metadata.dimensions == ("date", "stock")
        else (n_stocks,)
    )
    canonical_dtype = "utf8-nfc" if metadata.dtype == "str" else metadata.dtype
    digest = hashlib.sha256()
    descriptor = {
        "name": metadata.name,
        "dimensions": list(metadata.dimensions),
        "decision_lag": metadata.decision_lag,
        "source_dtype": values.dtype.str,
        "canonical_dtype": canonical_dtype,
        "canonical_shape": list(canonical_shape),
    }
    _update_framed(digest, _canonical_json(descriptor))

    selected = values[:prefix_rows] if values.ndim == 2 else values
    if metadata.dtype == "str":
        _hash_text_values(digest, selected)
    elif values.ndim == 1:
        digest.update(_canonical_numeric_chunk(selected, metadata.dtype).tobytes())
    else:
        for row_start in range(0, prefix_rows, _LINEAGE_CHUNK_ROWS):
            row_stop = min(prefix_rows, row_start + _LINEAGE_CHUNK_ROWS)
            normalized = _canonical_numeric_chunk(
                selected[row_start:row_stop],
                metadata.dtype,
            )
            digest.update(normalized.tobytes(order="C"))

    return RuntimeFieldLineage(
        name=metadata.name,
        dimensions=metadata.dimensions,
        decision_lag=metadata.decision_lag,
        source_dtype=values.dtype.str,
        canonical_dtype=canonical_dtype,
        canonical_shape=canonical_shape,
        semantic_sha256=digest.hexdigest(),
    )


def _date_axis_sha256(
    dates: np.ndarray,
    prefix_rows: int,
) -> str:
    prefix = np.array(
        dates[:prefix_rows].astype("datetime64[D]").view("<i8"),
        dtype="<i8",
        order="C",
        copy=True,
    )
    digest = hashlib.sha256()
    descriptor = {
        "name": "trade_dates",
        "source_dtype": dates.dtype.str,
        "canonical_dtype": "datetime64[D]",
        "canonical_shape": [prefix_rows],
    }
    _update_framed(digest, _canonical_json(descriptor))
    digest.update(prefix.tobytes(order="C"))
    return digest.hexdigest()


def _generation_semantics(
) -> tuple[str, tuple[RuntimeGenerationComponent, ...]]:
    repository_root = Path(__file__).resolve().parents[1]
    components = []
    for name, source_location in _GENERATION_SEMANTICS_COMPONENTS:
        source_path = Path(source_location)
        if not source_path.is_absolute():
            source_path = repository_root / source_path
        source = source_path.read_text(encoding="utf-8-sig")
        normalized = source.replace("\r\n", "\n").replace("\r", "\n")
        components.append(
            RuntimeGenerationComponent(
                name=name,
                source_sha256=hashlib.sha256(
                    normalized.encode("utf-8")
                ).hexdigest(),
            )
        )
    payload = {
        "version": RUNTIME_GENERATION_SEMANTICS_VERSION,
        "components": [component.as_dict() for component in components],
    }
    return (
        hashlib.sha256(_canonical_json(payload)).hexdigest(),
        tuple(components),
    )


def compute_runtime_lineage(
    npz_path: str | Path,
    cutoff: object | None = None,
) -> RuntimeLineage:
    """Compute a canonical semantic identity for a local runtime prefix.

    Date-stock fields contribute only rows through ``cutoff``. Stock fields
    always contribute their complete values. The hash therefore remains
    stable when a compatible runtime only appends dates after the cutoff.
    """

    source_path = Path(npz_path).resolve()
    with np.load(source_path, allow_pickle=False) as npz:
        dates = _validate_dates(npz["trade_dates"])
        stock_codes = _validate_stock_codes(npz["stock_codes"])
        fields = _selected_fields(npz.files)
        requested_cutoff = None if cutoff is None else _as_date(cutoff)
        prefix_rows = (
            len(dates)
            if requested_cutoff is None
            else int(np.searchsorted(dates, requested_cutoff, side="right"))
        )
        if prefix_rows == 0:
            raise ValueError("lineage cutoff precedes the first runtime date")

        schema_hash = _schema_hash(fields)
        generation_hash, generation_components = _generation_semantics()
        vocabulary_hash = _stock_vocabulary_hash(stock_codes)
        date_hash = _date_axis_sha256(dates, prefix_rows)
        field_lineages = []
        for metadata in fields:
            values = npz[metadata.name]
            field_lineages.append(
                _compute_field_lineage(
                    metadata,
                    values,
                    prefix_rows=prefix_rows,
                    n_dates=len(dates),
                    n_stocks=len(stock_codes),
                )
            )
            del values

    prefix_start = str(dates[0])
    prefix_end = str(dates[prefix_rows - 1])
    identity = {
        "lineage_version": RUNTIME_LINEAGE_VERSION,
        "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
        "runtime_schema_hash": schema_hash,
        "generation_semantics_version": RUNTIME_GENERATION_SEMANTICS_VERSION,
        "generation_semantics_sha256": generation_hash,
        "generation_components": [
            component.as_dict() for component in generation_components
        ],
        "prefix_start": prefix_start,
        "prefix_end": prefix_end,
        "prefix_rows": prefix_rows,
        "stock_count": len(stock_codes),
        "stock_vocabulary_sha256": vocabulary_hash,
        "date_axis_sha256": date_hash,
        "fields": [field.as_dict() for field in field_lineages],
    }
    return RuntimeLineage(
        lineage_version=RUNTIME_LINEAGE_VERSION,
        runtime_schema_version=RUNTIME_SCHEMA_VERSION,
        runtime_schema_hash=schema_hash,
        generation_semantics_version=RUNTIME_GENERATION_SEMANTICS_VERSION,
        generation_semantics_sha256=generation_hash,
        generation_components=generation_components,
        upstream_provenance_embedded=False,
        provenance_note=LEGACY_RUNTIME_PROVENANCE_NOTE,
        semantic_sha256=hashlib.sha256(_canonical_json(identity)).hexdigest(),
        requested_cutoff=(
            None if requested_cutoff is None else str(requested_cutoff)
        ),
        prefix_start=prefix_start,
        prefix_end=prefix_end,
        prefix_rows=prefix_rows,
        stock_count=len(stock_codes),
        stock_vocabulary_sha256=vocabulary_hash,
        date_axis_sha256=date_hash,
        fields=tuple(field_lineages),
    )


def load_runtime_slice(
    npz_path: str | Path,
    start: object,
    end: object,
    *,
    lookback: int = DEFAULT_LOOKBACK,
    max_factor_history: int = MAX_PRODUCTION_FACTOR_HISTORY,
    expected_stock_codes: Iterable[str] | None = None,
) -> RuntimeSlice:
    """Load one local NPZ into a sealed, copied runtime slice.

    No path discovery or network fallback occurs. ``end`` is a hard boundary;
    the result never includes a later row, including a validation/test row used
    only to settle an earlier split.
    """

    source_path = Path(npz_path).resolve()
    requested_start = _as_date(start)
    requested_end = _as_date(end)
    if requested_start > requested_end:
        raise ValueError("start must not be later than end")
    if lookback < 1 or max_factor_history < 0:
        raise ValueError("lookback must be positive and factor history non-negative")

    requested_preload = max(
        MIN_PRELOAD_ROWS,
        int(lookback) + int(max_factor_history) + 2,
    )

    with np.load(source_path, allow_pickle=False) as npz:
        source_dates = _validate_dates(npz["trade_dates"])
        source_codes = _validate_stock_codes(npz["stock_codes"])
        if expected_stock_codes is not None:
            expected = tuple(str(code) for code in expected_stock_codes)
            if source_codes != expected:
                raise ValueError("runtime stock vocabulary or order does not match")

        decision_start_source = int(
            np.searchsorted(source_dates, requested_start, side="left")
        )
        decision_stop_source = int(
            np.searchsorted(source_dates, requested_end, side="right")
        )
        if decision_start_source >= decision_stop_source:
            raise ValueError("requested range contains no runtime trading dates")

        row_start = max(0, decision_start_source - requested_preload)
        row_stop = decision_stop_source
        fields = _selected_fields(npz.files)
        n_dates = len(source_dates)
        n_stocks = len(source_codes)
        data = {
            metadata.name: _copy_field(
                npz[metadata.name],
                metadata,
                row_start,
                row_stop,
                n_dates,
                n_stocks,
            )
            for metadata in fields
        }
        trade_dates = _owned_read_only(
            source_dates[row_start:row_stop],
            dtype="datetime64[D]",
        )

    decision_start = decision_start_source - row_start
    decision_stop = decision_stop_source - row_start
    source_stat = source_path.stat()
    manifest = RuntimeManifest(
        schema_version=RUNTIME_SCHEMA_VERSION,
        schema_hash=_schema_hash(fields),
        source_path=str(source_path),
        source_sha256=file_sha256(source_path),
        source_size=source_stat.st_size,
        stock_vocabulary_sha256=_stock_vocabulary_hash(source_codes),
        requested_start=str(requested_start),
        requested_end=str(requested_end),
        loaded_start=str(trade_dates[0]),
        loaded_end=str(trade_dates[-1]),
        requested_preload_rows=requested_preload,
        actual_preload_rows=decision_start,
        fields=fields,
    )
    return RuntimeSlice(
        stock_codes=source_codes,
        trade_dates=trade_dates,
        data=data,
        decision_start=decision_start,
        decision_stop=decision_stop,
        manifest=manifest,
    )
