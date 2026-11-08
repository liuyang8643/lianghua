"""Strict identity and deployment checks for a frozen policy artifact."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping

from utils.atomic_file import atomic_write_json, file_sha256


BUNDLE_VERSION = "wbr-policy-bundle-v47-amihud12-hold20"
MANIFEST_FILE = "manifest.json"
DEPLOYMENT_GATE_NAMES = (
    "trained_checkpoint_selected",
    "finite",
    "full_investment_contract_satisfied",
    "validation_selected",
    "test_reported",
)


def source_tree_sha256(root: str | Path, paths: Iterable[str | Path]) -> str:
    base = Path(root).resolve()
    resolved = sorted(Path(path).resolve() for path in paths)
    digest = hashlib.sha256()
    for path in resolved:
        relative = path.relative_to(base).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def policy_source_sha256(repo_root: str | Path) -> str:
    """Hash every source file that can alter policy or environment semantics."""

    root = Path(repo_root).resolve()
    files: list[Path] = []
    for package in ("ai", "env", "factor", "offline_data"):
        files.extend((root / package).rglob("*.py"))
    files.extend(
        path
        for path in (
            root / "factor_db" / "factors" / "AmihudIlliquidity.py",
            root / "factor_db" / "factors" / "TrueMarketCap.py",
            root / "factor_db" / "factors" / "VolumeCV.py",
            root / "factor_db" / "factors" / "AmountBasedSmallCap.py",
            root / "factor_db" / "factors" / "filter.py",
            root / "utils" / "atomic_file.py",
            root / "utils" / "stable_sort.py",
            root / "configs" / "training.py",
        )
        if path.is_file()
    )
    return source_tree_sha256(root, files)


def _require_candidate_evaluation(
    split: str,
    summary: object,
) -> float:
    if not isinstance(summary, Mapping):
        raise ValueError(f"policy bundle has no {split} evaluation")
    if summary.get("full_investment_contract_satisfied") is not True:
        raise ValueError(f"policy bundle {split} lacks full-investment proof")
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"policy bundle {split} metrics are missing")
    try:
        calmar = float(metrics["calmar"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"policy bundle {split} Calmar is invalid") from exc
    if not math.isfinite(calmar):
        raise ValueError(f"policy bundle {split} Calmar is non-finite")
    return calmar


@dataclass(frozen=True)
class BundleManifest:
    created_at: str
    algorithm: str
    model_file: str
    model_sha256: str
    normalizer_file: str
    normalizer_sha256: str
    config_file: str
    config_sha256: str
    source_sha256: str
    runtime: Mapping[str, object]
    factors: Mapping[str, object]
    observation_schema: Mapping[str, object]
    encoded_schema: Mapping[str, object]
    action_schema: Mapping[str, object]
    environment: Mapping[str, object]
    training: Mapping[str, object]
    evaluation: Mapping[str, object]
    bundle_version: str = BUNDLE_VERSION

    def __post_init__(self) -> None:
        if self.bundle_version != BUNDLE_VERSION:
            raise ValueError(f"unsupported bundle version: {self.bundle_version}")
        for name in (
            "model_sha256",
            "normalizer_sha256",
            "config_sha256",
            "source_sha256",
        ):
            value = str(getattr(self, name))
            if len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 digest")
        for name in (
            "model_file",
            "normalizer_file",
            "config_file",
            "created_at",
            "algorithm",
        ):
            if not str(getattr(self, name)):
                raise ValueError(f"{name} must not be empty")
        for name in (
            "runtime",
            "factors",
            "observation_schema",
            "encoded_schema",
            "action_schema",
            "environment",
            "training",
            "evaluation",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    def to_dict(self) -> dict[str, object]:
        return {
            "bundle_version": self.bundle_version,
            "created_at": self.created_at,
            "algorithm": self.algorithm,
            "model_file": self.model_file,
            "model_sha256": self.model_sha256,
            "normalizer_file": self.normalizer_file,
            "normalizer_sha256": self.normalizer_sha256,
            "config_file": self.config_file,
            "config_sha256": self.config_sha256,
            "source_sha256": self.source_sha256,
            "runtime": dict(self.runtime),
            "factors": dict(self.factors),
            "observation_schema": dict(self.observation_schema),
            "encoded_schema": dict(self.encoded_schema),
            "action_schema": dict(self.action_schema),
            "environment": dict(self.environment),
            "training": dict(self.training),
            "evaluation": dict(self.evaluation),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "BundleManifest":
        version = str(payload.get("bundle_version", ""))
        if version != BUNDLE_VERSION:
            raise ValueError(f"unsupported bundle version: {version}")
        return cls(
            bundle_version=version,
            created_at=str(payload["created_at"]),
            algorithm=str(payload["algorithm"]),
            model_file=str(payload["model_file"]),
            model_sha256=str(payload["model_sha256"]),
            normalizer_file=str(payload["normalizer_file"]),
            normalizer_sha256=str(payload["normalizer_sha256"]),
            config_file=str(payload["config_file"]),
            config_sha256=str(payload["config_sha256"]),
            source_sha256=str(payload["source_sha256"]),
            runtime=dict(payload["runtime"]),
            factors=dict(payload["factors"]),
            observation_schema=dict(payload["observation_schema"]),
            encoded_schema=dict(payload["encoded_schema"]),
            action_schema=dict(payload["action_schema"]),
            environment=dict(payload["environment"]),
            training=dict(payload["training"]),
            evaluation=dict(payload["evaluation"]),
        )

    def save(self, directory: str | Path) -> Path:
        target = Path(directory) / MANIFEST_FILE
        atomic_write_json(target, self.to_dict())
        return target

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        verify_files: bool = True,
    ) -> "BundleManifest":
        root = Path(directory)
        payload = json.loads((root / MANIFEST_FILE).read_text(encoding="utf-8"))
        manifest = cls.from_dict(payload)
        if verify_files:
            for file_name, expected in (
                (manifest.model_file, manifest.model_sha256),
                (manifest.normalizer_file, manifest.normalizer_sha256),
                (manifest.config_file, manifest.config_sha256),
            ):
                if file_sha256(root / file_name) != expected:
                    raise ValueError(f"bundle file hash mismatch: {file_name}")
        return manifest

    @property
    def technical_convergence(self) -> bool:
        convergence = self.training.get("convergence")
        return bool(
            isinstance(convergence, Mapping)
            and convergence.get("technical_convergence") is True
        )

    def require_deployable(self) -> None:
        convergence = self.training.get("convergence")
        if not isinstance(convergence, Mapping):
            raise ValueError("policy bundle has no convergence record")
        failed = [
            name
            for name in DEPLOYMENT_GATE_NAMES
            if convergence.get(name) is not True
        ]
        if failed or not self.technical_convergence:
            raise ValueError(
                "policy bundle is diagnostic-only: "
                + ", ".join(failed or ["technical_convergence"])
            )
        for split in ("train", "validation", "test"):
            _require_candidate_evaluation(split, self.evaluation.get(split))
        selection = self.training.get("checkpoint_selection")
        if not isinstance(selection, Mapping) or (
            selection.get("objective") != "full_validation_calmar"
        ):
            raise ValueError("policy bundle was not selected by validation Calmar")
        if not self.environment:
            raise ValueError("policy bundle has no frozen environment semantics")
        financial = self.runtime.get("financial_snapshot")
        if not isinstance(financial, Mapping):
            raise ValueError("policy bundle has no financial snapshot provenance")
        for name in ("manifest_sha256", "snapshot_sha256", "financial_identity_sha256"):
            value = financial.get(name)
            if not isinstance(value, str) or len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"policy bundle financial snapshot has invalid {name}")
        for name in ("panel_builder_version", "financial_replay_version", "availability", "pit_evidence_limit"):
            if not isinstance(financial.get(name), str) or not financial[name]:
                raise ValueError(f"policy bundle financial snapshot lacks {name}")
        lineage = self.runtime.get("lineage")
        required_lineage = {
            "lineage_version",
            "runtime_schema_hash",
            "generation_semantics_sha256",
            "semantic_sha256",
            "prefix_start",
            "prefix_end",
            "prefix_rows",
            "stock_vocabulary_sha256",
            "date_axis_sha256",
            "fields",
        }
        if not isinstance(lineage, Mapping):
            raise ValueError("policy bundle has no runtime prefix lineage")
        missing = required_lineage - set(lineage)
        if missing:
            raise ValueError(
                "policy bundle has incomplete runtime prefix lineage: "
                + ", ".join(sorted(missing))
            )


__all__ = [
    "BUNDLE_VERSION",
    "DEPLOYMENT_GATE_NAMES",
    "BundleManifest",
    "file_sha256",
    "policy_source_sha256",
    "source_tree_sha256",
]
