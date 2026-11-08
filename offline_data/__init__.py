"""Public offline data contracts and sealed runtime loader."""

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
from .runtime import (
    OPTIONAL_RUNTIME_FIELDS,
    RUNTIME_FIELDS,
    compute_runtime_lineage,
    file_sha256,
    latest_runtime_npz_path,
    load_runtime_stock_codes,
    load_runtime_slice,
)
from .identity import validate_runtime_identity, validate_runtime_path_identity
from .financial_versions import ABNORMAL_GROSS_PROFIT_PANEL_FIELDS, FINANCIAL_PANEL_FIELDS

__all__ = [
    "ABNORMAL_GROSS_PROFIT_PANEL_FIELDS",
    "FINANCIAL_PANEL_FIELDS",
    "OPTIONAL_RUNTIME_FIELDS",
    "LEGACY_RUNTIME_PROVENANCE_NOTE",
    "RUNTIME_GENERATION_SEMANTICS_VERSION",
    "RUNTIME_FIELDS",
    "RUNTIME_LINEAGE_VERSION",
    "RUNTIME_SCHEMA_VERSION",
    "RuntimeFieldMetadata",
    "RuntimeFieldLineage",
    "RuntimeGenerationComponent",
    "RuntimeLineage",
    "RuntimeManifest",
    "RuntimeSlice",
    "file_sha256",
    "latest_runtime_npz_path",
    "load_runtime_stock_codes",
    "compute_runtime_lineage",
    "load_runtime_slice",
    "validate_runtime_identity",
    "validate_runtime_path_identity",
]
