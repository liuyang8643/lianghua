from __future__ import annotations

import hashlib

import numpy as np
import pytest

import offline_data.runtime as runtime_module
from offline_data import (
    MIN_PRELOAD_ROWS,
    RUNTIME_LINEAGE_VERSION,
    compute_runtime_lineage,
    load_runtime_slice,
)


def _runtime_arrays(rows: int = 180, stocks: int = 3) -> dict[str, np.ndarray]:
    dates = np.datetime64("2020-01-01") + np.arange(rows)
    day = np.arange(rows, dtype=np.float64)[:, None]
    stock = np.arange(stocks, dtype=np.float64)[None, :]
    base = 10.0 + day * 0.01 + stock
    data = {
        "trade_dates": dates.astype("datetime64[D]"),
        "stock_codes": np.array(
            [f"{index + 1:06d}.SZ" for index in range(stocks)]
        ),
        "open": base,
        "high": base + 0.5,
        "low": base - 0.5,
        "close": base + 0.1,
        "volume": 1_000_000.0 + day * 1000.0 + stock * 100.0,
        "amount": 100_000_000.0 + day * 100_000.0 + stock * 10_000.0,
        "preClose": base - 0.1,
        "total_share": 100_000_000.0 + day * 10_000.0 + stock * 1000.0,
        "bps": np.full((rows, stocks), 3.0),
        "eps": np.full((rows, stocks), 0.5),
        "roe": np.full((rows, stocks), 0.1),
        "profit_yoy": np.full((rows, stocks), 0.2),
        "revenue_yoy": np.full((rows, stocks), 0.15),
        "operating_cf_ps": np.full((rows, stocks), 0.4),
        "gross_margin": np.full((rows, stocks), 0.3),
        "st_mask": np.zeros((rows, stocks), dtype=np.bool_),
        "issue_price": np.linspace(5.0, 7.0, stocks),
        "stock_names": np.array([f"stock-{index}" for index in range(stocks)]),
    }
    return data


def _write_runtime(path, **changes) -> dict[str, np.ndarray]:
    data = _runtime_arrays()
    data.update(changes)
    np.savez(path, **data)
    return data


def test_runtime_slice_is_strict_copied_contiguous_and_sealed(tmp_path):
    path = tmp_path / "runtime.npz"
    source = _write_runtime(path)
    start = source["trade_dates"][150]
    end = source["trade_dates"][160]

    runtime = load_runtime_slice(path, start, end)

    assert runtime.decision_start == MIN_PRELOAD_ROWS
    assert runtime.decision_stop - runtime.decision_start == 11
    assert runtime.trade_dates[0] == source["trade_dates"][24]
    assert runtime.trade_dates[-1] == end
    assert np.all(runtime.trade_dates <= end)
    assert runtime.manifest.requested_preload_rows == MIN_PRELOAD_ROWS
    assert runtime.manifest.actual_preload_rows == MIN_PRELOAD_ROWS
    assert runtime.manifest.loaded_end == str(end)
    assert runtime.manifest.source_sha256 == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    assert runtime.trade_dates.base is None
    assert runtime.trade_dates.flags.owndata
    assert not runtime.trade_dates.flags.writeable
    for values in runtime.data.values():
        assert values.base is None
        assert values.flags.owndata
        assert values.flags.c_contiguous
        assert not values.flags.writeable

    assert runtime.field("open").dtype == np.float32
    assert runtime.field("st_mask").dtype == np.bool_
    assert runtime.field("issue_price").shape == (3,)
    assert runtime.field("stock_names").shape == (3,)
    assert runtime.index_of(start) == runtime.decision_start


def test_runtime_slice_never_crosses_training_end(tmp_path):
    path = tmp_path / "runtime.npz"
    source = _write_runtime(path)
    train_end = source["trade_dates"][130]
    first_validation_date = source["trade_dates"][131]

    runtime = load_runtime_slice(
        path,
        source["trade_dates"][120],
        train_end,
    )

    assert runtime.trade_dates[-1] == train_end
    assert first_validation_date not in runtime.trade_dates
    with pytest.raises(KeyError, match="not present"):
        runtime.index_of(first_validation_date)


def test_runtime_slice_validates_date_and_stock_axes(tmp_path):
    path = tmp_path / "runtime.npz"
    source = _runtime_arrays()
    source["trade_dates"][[10, 11]] = source["trade_dates"][[11, 10]]
    np.savez(path, **source)

    with pytest.raises(ValueError, match="strictly increasing"):
        load_runtime_slice(path, "2020-05-01", "2020-05-10")

    nat_path = tmp_path / "runtime-nat.npz"
    source = _runtime_arrays()
    source["trade_dates"][10] = np.datetime64("NaT")
    np.savez(nat_path, **source)
    with pytest.raises(ValueError, match="must not contain NaT"):
        compute_runtime_lineage(nat_path)

    path = tmp_path / "runtime-valid.npz"
    source = _write_runtime(path)
    with pytest.raises(ValueError, match="vocabulary or order"):
        load_runtime_slice(
            path,
            source["trade_dates"][130],
            source["trade_dates"][140],
            expected_stock_codes=reversed(source["stock_codes"]),
        )


def test_runtime_slice_uses_available_history_at_source_start(tmp_path):
    path = tmp_path / "runtime.npz"
    source = _write_runtime(path)

    runtime = load_runtime_slice(
        path,
        source["trade_dates"][20],
        source["trade_dates"][25],
    )

    assert runtime.decision_start == 20
    assert runtime.manifest.actual_preload_rows == 20
    assert runtime.trade_dates[0] == source["trade_dates"][0]


def test_runtime_slice_rejects_unregistered_dimensions(tmp_path):
    path = tmp_path / "runtime.npz"
    source = _runtime_arrays()
    source["new_market_field"] = np.ones_like(source["open"])
    np.savez(path, **source)

    with pytest.raises(ValueError, match="unregistered fields"):
        load_runtime_slice(path, "2020-05-01", "2020-05-10")


def _copy_runtime(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: values.copy() for name, values in data.items()}


def test_runtime_lineage_is_semantic_not_npz_container_identity(tmp_path):
    data = _runtime_arrays(rows=8, stocks=2)
    plain = tmp_path / "plain.npz"
    compressed = tmp_path / "compressed.npz"
    np.savez(plain, **data)
    np.savez_compressed(compressed, **data)

    first = compute_runtime_lineage(plain)
    second = compute_runtime_lineage(compressed)

    assert first.lineage_version == RUNTIME_LINEAGE_VERSION
    assert first.prefix_sha256 == second.prefix_sha256
    assert first.date_axis_sha256 == second.date_axis_sha256
    assert first.fields == second.fields
    assert first.prefix_rows == 8
    assert first.field("open").canonical_shape == (8, 2)
    assert {component.name for component in first.generation_components} >= {
        "runtime_builder",
        "financial_pit",
    }
    assert not first.upstream_provenance_embedded
    assert "do not embed upstream source provenance" in first.provenance_note


def test_runtime_lineage_allows_only_post_cutoff_append(tmp_path):
    base = _runtime_arrays(rows=8, stocks=2)
    appended = _runtime_arrays(rows=11, stocks=2)
    base_path = tmp_path / "base.npz"
    appended_path = tmp_path / "appended.npz"
    np.savez(base_path, **base)
    np.savez(appended_path, **appended)
    cutoff = base["trade_dates"][-1]

    base_prefix = compute_runtime_lineage(base_path, cutoff=cutoff)
    appended_prefix = compute_runtime_lineage(appended_path, cutoff=cutoff)
    appended_full = compute_runtime_lineage(appended_path)

    assert base_prefix.prefix_sha256 == appended_prefix.prefix_sha256
    assert base_prefix.date_axis_sha256 == appended_prefix.date_axis_sha256
    assert base_prefix.fields == appended_prefix.fields
    assert base_prefix.prefix_rows == appended_prefix.prefix_rows == 8
    assert appended_full.prefix_sha256 != base_prefix.prefix_sha256


def test_runtime_lineage_detects_historical_market_and_financial_rewrites(
    tmp_path,
):
    base = _runtime_arrays(rows=8, stocks=2)
    changed = _copy_runtime(base)
    changed["open"][3, 0] += 0.25
    changed["roe"][4, 1] += 0.5
    base_path = tmp_path / "base.npz"
    changed_path = tmp_path / "changed.npz"
    np.savez(base_path, **base)
    np.savez(changed_path, **changed)

    baseline = compute_runtime_lineage(base_path)
    rewritten = compute_runtime_lineage(changed_path)

    assert baseline.prefix_sha256 != rewritten.prefix_sha256
    assert baseline.date_axis_sha256 == rewritten.date_axis_sha256
    assert (
        baseline.field("open").semantic_sha256
        != rewritten.field("open").semantic_sha256
    )
    assert (
        baseline.field("roe").semantic_sha256
        != rewritten.field("roe").semantic_sha256
    )
    assert baseline.field("close") == rewritten.field("close")


def test_runtime_lineage_detects_date_rewrite_and_insertion(tmp_path):
    base = _runtime_arrays(rows=6, stocks=2)
    base["trade_dates"] = (
        np.datetime64("2020-01-01") + 2 * np.arange(6)
    ).astype("datetime64[D]")
    rewritten = _copy_runtime(base)
    rewritten["trade_dates"][2] -= np.timedelta64(1, "D")

    inserted = {}
    for name, values in base.items():
        if name == "trade_dates":
            inserted[name] = np.insert(
                values,
                3,
                np.datetime64("2020-01-06"),
            )
        elif values.ndim == 2:
            inserted[name] = np.insert(values, 3, values[2], axis=0)
        else:
            inserted[name] = values.copy()

    base_path = tmp_path / "base.npz"
    rewritten_path = tmp_path / "rewritten.npz"
    inserted_path = tmp_path / "inserted.npz"
    np.savez(base_path, **base)
    np.savez(rewritten_path, **rewritten)
    np.savez(inserted_path, **inserted)
    baseline = compute_runtime_lineage(base_path)
    date_rewrite = compute_runtime_lineage(rewritten_path)
    date_insert = compute_runtime_lineage(
        inserted_path,
        cutoff=base["trade_dates"][-1],
    )

    assert baseline.date_axis_sha256 != date_rewrite.date_axis_sha256
    assert baseline.prefix_sha256 != date_rewrite.prefix_sha256
    assert baseline.field("open") == date_rewrite.field("open")
    assert baseline.date_axis_sha256 != date_insert.date_axis_sha256
    assert baseline.prefix_sha256 != date_insert.prefix_sha256
    assert date_insert.prefix_rows == baseline.prefix_rows + 1


@pytest.mark.parametrize("field", ["issue_price", "stock_names"])
def test_runtime_lineage_detects_stock_field_rewrite(tmp_path, field):
    base = _runtime_arrays(rows=8, stocks=2)
    changed = _copy_runtime(base)
    if field == "issue_price":
        changed[field][0] += 1.0
    else:
        changed[field][0] = "renamed"
    base_path = tmp_path / f"base-{field}.npz"
    changed_path = tmp_path / f"changed-{field}.npz"
    np.savez(base_path, **base)
    np.savez(changed_path, **changed)

    baseline = compute_runtime_lineage(base_path, cutoff=base["trade_dates"][4])
    rewritten = compute_runtime_lineage(
        changed_path,
        cutoff=base["trade_dates"][4],
    )

    assert baseline.prefix_sha256 != rewritten.prefix_sha256
    assert (
        baseline.field(field).semantic_sha256
        != rewritten.field(field).semantic_sha256
    )


def test_runtime_lineage_covers_source_dtype_and_registered_schema(tmp_path):
    base = _runtime_arrays(rows=8, stocks=2)
    dtype_changed = _copy_runtime(base)
    dtype_changed["roe"] = dtype_changed["roe"].astype(np.float32)
    schema_changed = _copy_runtime(base)
    schema_changed["star_st_mask"] = np.zeros_like(base["st_mask"])
    base_path = tmp_path / "base.npz"
    dtype_path = tmp_path / "dtype.npz"
    schema_path = tmp_path / "schema.npz"
    np.savez(base_path, **base)
    np.savez(dtype_path, **dtype_changed)
    np.savez(schema_path, **schema_changed)

    baseline = compute_runtime_lineage(base_path)
    dtype_lineage = compute_runtime_lineage(dtype_path)
    schema_lineage = compute_runtime_lineage(schema_path)

    assert baseline.field("roe").canonical_dtype == "float32"
    assert baseline.field("roe").source_dtype != dtype_lineage.field("roe").source_dtype
    assert baseline.prefix_sha256 != dtype_lineage.prefix_sha256
    assert baseline.runtime_schema_hash != schema_lineage.runtime_schema_hash
    assert baseline.prefix_sha256 != schema_lineage.prefix_sha256


def test_generation_semantics_changes_lineage_without_hashing_source_path(
    tmp_path,
    monkeypatch,
):
    data = _runtime_arrays(rows=8, stocks=2)
    runtime_path = tmp_path / "runtime.npz"
    source_a = tmp_path / "builder-a.py"
    source_b = tmp_path / "builder-b.py"
    np.savez(runtime_path, **data)
    source_a.write_text("ALIGNMENT_VERSION = 1\n", encoding="utf-8")
    source_b.write_text("ALIGNMENT_VERSION = 1\n", encoding="utf-8")

    monkeypatch.setattr(
        runtime_module,
        "_GENERATION_SEMANTICS_COMPONENTS",
        (("runtime_builder", source_a),),
    )
    baseline = compute_runtime_lineage(runtime_path)
    monkeypatch.setattr(
        runtime_module,
        "_GENERATION_SEMANTICS_COMPONENTS",
        (("runtime_builder", source_b),),
    )
    same_source_at_another_path = compute_runtime_lineage(runtime_path)
    source_b.write_text("ALIGNMENT_VERSION = 2\n", encoding="utf-8")
    changed = compute_runtime_lineage(runtime_path)

    assert (
        baseline.generation_semantics_sha256
        == same_source_at_another_path.generation_semantics_sha256
    )
    assert baseline.prefix_sha256 == same_source_at_another_path.prefix_sha256
    assert (
        baseline.generation_semantics_sha256
        != changed.generation_semantics_sha256
    )
    assert baseline.prefix_sha256 != changed.prefix_sha256
    assert baseline.date_axis_sha256 == changed.date_axis_sha256
    assert baseline.fields == changed.fields
