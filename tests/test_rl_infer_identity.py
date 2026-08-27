from types import SimpleNamespace

import pytest

from ai.rl.infer import _validate_runtime_identity


def _lineage(**changes):
    values = {
        "lineage_version": "wbr.runtime-lineage.v1",
        "runtime_schema_hash": "1" * 64,
        "generation_semantics_sha256": "3" * 64,
        "semantic_sha256": "4" * 64,
        "prefix_start": "1990-12-19",
        "prefix_end": "2026-08-27",
        "prefix_rows": 8_000,
        "stock_vocabulary_sha256": "2" * 64,
        "date_axis_sha256": "5" * 64,
        "fields": [],
    }
    values.update(changes)
    return values


def _trained(**changes):
    values = {
        "schema_hash": "1" * 64,
        "stock_vocabulary_sha256": "2" * 64,
        "source_sha256": "6" * 64,
        "lineage": _lineage(),
    }
    values.update(changes)
    return values


def _current(*, source_sha256: str = "6" * 64):
    return SimpleNamespace(
        schema_hash="1" * 64,
        stock_vocabulary_sha256="2" * 64,
        source_sha256=source_sha256,
    )


def test_runtime_identity_fast_path_accepts_the_exact_training_file():
    _validate_runtime_identity(_current(), _trained())


def test_runtime_identity_accepts_changed_file_only_with_same_prefix_lineage():
    _validate_runtime_identity(
        _current(source_sha256="7" * 64),
        _trained(),
        _lineage(),
    )


def test_runtime_identity_rejects_changed_file_without_prefix_verification():
    with pytest.raises(ValueError, match="append-only prefix verification"):
        _validate_runtime_identity(
            _current(source_sha256="7" * 64),
            _trained(),
        )


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("semantic_sha256", "historical content prefix"),
        ("date_axis_sha256", "historical date axis"),
        ("generation_semantics_sha256", "generation semantics"),
        ("prefix_rows", "prefix row count"),
    ),
)
def test_runtime_identity_rejects_rewritten_prefix_or_generation_semantics(
    field,
    message,
):
    changed = _lineage(**{field: "changed"})
    with pytest.raises(ValueError, match=message):
        _validate_runtime_identity(
            _current(source_sha256="7" * 64),
            _trained(),
            changed,
        )


def test_runtime_identity_requires_lineage_even_for_exact_source_file():
    trained = _trained()
    trained.pop("lineage")
    with pytest.raises(ValueError, match="no runtime prefix lineage"):
        _validate_runtime_identity(_current(), trained)


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("schema_hash", "runtime schema"),
        ("stock_vocabulary_sha256", "stock vocabulary"),
    ),
)
def test_runtime_identity_rejects_schema_or_vocabulary_drift(field, message):
    trained = _trained()
    trained[field] = "8" * 64

    with pytest.raises(ValueError, match=message):
        _validate_runtime_identity(_current(), trained)
