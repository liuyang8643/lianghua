"""Public append-only runtime identity verification shared by infer and trade."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from offline_data.contracts import RuntimeManifest
from offline_data.runtime import compute_runtime_lineage


def validate_runtime_identity(
    current: RuntimeManifest,
    trained: Mapping[str, object],
    current_lineage: Mapping[str, object] | None = None,
) -> None:
    """Allow only a semantically identical trained prefix plus new dates."""

    if current.schema_hash != str(trained["schema_hash"]):
        raise ValueError("runtime schema differs from the trained bundle")
    if current.stock_vocabulary_sha256 != str(
        trained["stock_vocabulary_sha256"]
    ):
        raise ValueError("runtime stock vocabulary/order differs from the trained bundle")
    trained_lineage = trained.get("lineage")
    if not isinstance(trained_lineage, Mapping):
        raise ValueError("trained bundle has no runtime prefix lineage")
    if str(trained_lineage.get("runtime_schema_hash")) != str(
        trained["schema_hash"]
    ):
        raise ValueError("trained runtime lineage conflicts with runtime schema")
    if str(trained_lineage.get("stock_vocabulary_sha256")) != str(
        trained["stock_vocabulary_sha256"]
    ):
        raise ValueError("trained runtime lineage conflicts with stock vocabulary")

    if current.source_sha256 == str(trained["source_sha256"]):
        return
    if current_lineage is None:
        raise ValueError(
            "changed runtime file requires append-only prefix verification"
        )
    comparisons = (
        ("lineage_version", "runtime lineage version"),
        ("runtime_schema_hash", "runtime lineage schema"),
        ("generation_semantics_sha256", "runtime generation semantics"),
        ("prefix_start", "runtime prefix start"),
        ("prefix_end", "runtime prefix end"),
        ("prefix_rows", "runtime prefix row count"),
        ("stock_vocabulary_sha256", "runtime lineage stock vocabulary"),
        ("date_axis_sha256", "runtime historical date axis"),
        ("semantic_sha256", "runtime historical content prefix"),
    )
    for field, label in comparisons:
        if current_lineage.get(field) != trained_lineage.get(field):
            raise ValueError(f"{label} differs from the trained bundle")


def validate_runtime_path_identity(
    current: RuntimeManifest,
    trained: Mapping[str, object],
    runtime_path: str | Path,
) -> None:
    """Verify exact bytes or recompute the frozen prefix proof for an append."""

    current_lineage = None
    if current.source_sha256 != str(trained["source_sha256"]):
        trained_lineage = trained.get("lineage")
        if not isinstance(trained_lineage, Mapping):
            raise ValueError("trained bundle has no runtime prefix lineage")
        cutoff = trained_lineage.get("prefix_end")
        if not isinstance(cutoff, str) or not cutoff:
            raise ValueError("trained runtime lineage has no prefix cutoff")
        current_lineage = compute_runtime_lineage(
            runtime_path,
            cutoff=cutoff,
        ).as_dict()
    validate_runtime_identity(current, trained, current_lineage)


__all__ = ["validate_runtime_identity", "validate_runtime_path_identity"]
