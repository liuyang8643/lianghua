"""Strict JSON identity for a frozen policy artifact."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping


BUNDLE_VERSION = "wbr-policy-bundle-v5"
MANIFEST_FILE = "manifest.json"
DEPLOYMENT_GATE_NAMES = (
    "trained_checkpoint_selected",
    "finite",
    "beats_untrained_threshold",
    "beats_random_threshold",
    "beats_fixed_config_threshold",
    "validation_noninferiority_and_positive_return",
    "plateau_reached",
    "average_exposure_at_least_0_45",
    "continuous_parameter_dynamic",
    "binary_parameter_dynamic",
    "discrete_parameter_dynamic",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_sha256(root: str | Path, paths: Iterable[str | Path]) -> str:
    base = Path(root).resolve()
    resolved = sorted(Path(path).resolve() for path in paths)
    digest = hashlib.sha256()
    for path in resolved:
        relative = path.relative_to(base).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        # Git working trees may expose the same Python source with LF, CRLF,
        # or mixed legacy line endings. Source identity is semantic across
        # platforms, so canonicalize line endings before hashing.
        content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def policy_source_sha256(repo_root: str | Path) -> str:
    """Hash the code that defines training, inference, and env semantics.

    This is intentionally independent of generated artifacts and repository
    metadata.  A frozen bundle must not silently run against changed action,
    observation, planner, settlement, or PPO assembly code.
    """

    root = Path(repo_root).resolve()
    files: list[Path] = []
    for package in ("ai", "env", "factor", "offline_data"):
        files.extend((root / package).rglob("*.py"))
    files.extend(
        (
            root / "factor_db" / "factors" / "AmihudIlliquidity.py",
            root / "factor_db" / "factors" / "TrueMarketCap.py",
            root / "factor_db" / "factors" / "VolumeCV.py",
            root / "factor_db" / "factors" / "AmountBasedSmallCap.py",
            root / "factor_db" / "factors" / "filter.py",
        )
    )
    return source_tree_sha256(root, files)


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
        bundle_version = str(payload.get("bundle_version", ""))
        if bundle_version != BUNDLE_VERSION:
            raise ValueError(f"unsupported bundle version: {bundle_version}")
        return cls(
            bundle_version=bundle_version,
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
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, directory: str | Path, *, verify_files: bool = True) -> "BundleManifest":
        root = Path(directory)
        payload = json.loads((root / MANIFEST_FILE).read_text(encoding="utf-8"))
        manifest = cls.from_dict(payload)
        if verify_files:
            for file_name, expected in (
                (manifest.model_file, manifest.model_sha256),
                (manifest.normalizer_file, manifest.normalizer_sha256),
                (manifest.config_file, manifest.config_sha256),
            ):
                actual = file_sha256(root / file_name)
                if actual != expected:
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
        if not self.technical_convergence:
            raise ValueError("policy bundle is diagnostic-only: convergence gate failed")
        convergence = self.training["convergence"]
        failed = [name for name in DEPLOYMENT_GATE_NAMES if convergence.get(name) is not True]
        if failed:
            raise ValueError(
                "policy bundle has incomplete deployment gates: " + ", ".join(failed)
            )
        missing = {"train", "validation", "test"} - set(self.evaluation)
        if missing:
            raise ValueError(
                "policy bundle is diagnostic-only: missing sealed evaluation splits "
                + ", ".join(sorted(missing))
            )
        if not self.environment:
            raise ValueError("policy bundle has no frozen environment semantics")
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
        missing_lineage = required_lineage - set(lineage)
        if missing_lineage:
            raise ValueError(
                "policy bundle has incomplete runtime prefix lineage: "
                + ", ".join(sorted(missing_lineage))
            )


__all__ = [
    "BUNDLE_VERSION",
    "DEPLOYMENT_GATE_NAMES",
    "BundleManifest",
    "file_sha256",
    "policy_source_sha256",
    "source_tree_sha256",
]
