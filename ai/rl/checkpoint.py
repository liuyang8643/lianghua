"""Canonical atomic persistence for the one resumable PPO checkpoint."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile

from ai.bundle import file_sha256
from utils.atomic_file import atomic_write_json, replace_file


LATEST_TRAIN_CHECKPOINT_VERSION = "wbr-latest-train-checkpoint-v2-standard-ppo"
LATEST_TRAIN_MODEL_FILE = "latest_train_model.zip"
LATEST_TRAIN_STATE_FILE = "latest_train_model.json"
EVALUATION_CHECKPOINT_VERSION = "wbr-evaluation-checkpoint-v1-calmar-tracking"


def _evaluation_state(*, checkpoint: Path, sha256: str, identity_version: str,
                      identity: str, timesteps: int, updates: int,
                      role: str, metrics: dict[str, float]) -> dict[str, object]:
    return {
        "schema_version": EVALUATION_CHECKPOINT_VERSION,
        "file": checkpoint.name,
        "sha256": sha256,
        "run_identity_version": identity_version,
        "run_identity_sha256": identity,
        "timesteps": timesteps,
        "ppo_updates": updates,
        "role": role,
        "metrics": dict(metrics),
    }


@dataclass(frozen=True)
class FrozenEvaluationCheckpoint:
    path: Path
    sha256: str
    identity_version: str
    identity: str
    timesteps: int
    updates: int

    def sidecar(self, *, target: Path, role: str, metrics: dict[str, float]) -> dict[str, object]:
        return _evaluation_state(checkpoint=target, sha256=self.sha256,
                                 identity_version=self.identity_version, identity=self.identity,
                                 timesteps=self.timesteps, updates=self.updates, role=role, metrics=metrics)

    def validate(self) -> None:
        expected = self.sidecar(target=self.path, role="pending_frozen_evaluation", metrics={})
        actual = json.loads(self.path.with_suffix(".json").read_text(encoding="utf-8"))
        if actual != expected or file_sha256(self.path) != self.sha256:
            raise ValueError("frozen evaluation checkpoint or sidecar changed")

    def release(self) -> None:
        self.path.unlink()
        self.path.with_suffix(".json").unlink()


def capture_evaluation_checkpoint(model, output_dir: Path) -> FrozenEvaluationCheckpoint:
    directory = output_dir / "evaluation_snapshots"
    name = f"candidate_{int(model.num_timesteps):012d}_{int(model._n_updates):08d}.zip"
    path = directory / name
    if path.exists() or path.with_suffix(".json").exists():
        raise ValueError("frozen evaluation snapshot already exists")
    state = save_evaluation_checkpoint(model, directory, file_name=name,
                                      run_identity_sha256=str(model.wbr_run_identity_sha256),
                                      role="pending_frozen_evaluation")
    snapshot = FrozenEvaluationCheckpoint(path, str(state["sha256"]),
                                      str(state["run_identity_version"]), str(state["run_identity_sha256"]),
                                      int(state["timesteps"]), int(state["ppo_updates"]))
    archive = output_dir / "checkpoint_archive"
    archive.mkdir(exist_ok=True)
    promote_evaluation_checkpoint(
        snapshot, archive, file_name=f"{path.stem}_{snapshot.sha256[:16]}.zip",
        role="diagnostic_archive_not_canonical_resume", metrics={},
    )
    return snapshot


def promote_evaluation_checkpoint(snapshot: FrozenEvaluationCheckpoint, output_dir: Path, *,
                                 file_name: str, role: str, metrics: dict[str, float]) -> dict[str, object]:
    if Path(file_name).name != file_name or not file_name.endswith(".zip") or not role:
        raise ValueError("evaluation destination must be a local zip with a role")
    snapshot.validate()
    target = output_dir / file_name
    descriptor, name = tempfile.mkstemp(prefix=f".{target.stem}-", suffix=".tmp.zip", dir=output_dir)
    os.close(descriptor)
    temporary = Path(name)
    try:
        shutil.copyfile(snapshot.path, temporary)
        if file_sha256(temporary) != snapshot.sha256:
            raise ValueError("frozen evaluation checkpoint changed while copying")
        replace_file(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    state = snapshot.sidecar(target=target, role=role, metrics=metrics)
    atomic_write_json(target.with_suffix(".json"), state)
    return state


def atomic_save_model(model, target: Path) -> Path:
    checkpoint = target if target.suffix == ".zip" else target.with_suffix(".zip")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{checkpoint.stem}-",
        suffix=".tmp.zip",
        dir=checkpoint.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        model.save(temporary)
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RuntimeError("SB3 did not create a complete temporary checkpoint")
        replace_file(temporary, checkpoint)
    finally:
        temporary.unlink(missing_ok=True)
    return checkpoint


def save_latest_train_checkpoint(
    model,
    output_dir: Path,
    *,
    run_identity_sha256: str,
    timesteps: int,
    ppo_updates: int,
    phase: str,
) -> dict[str, object]:
    if len(run_identity_sha256) != 64:
        raise ValueError("latest checkpoint run identity must be a SHA-256 digest")
    if timesteps < 0 or ppo_updates < 0:
        raise ValueError("latest checkpoint counters must be non-negative")
    if int(getattr(model, "num_timesteps", -1)) != timesteps or int(
        getattr(model, "_n_updates", -1)
    ) != ppo_updates:
        raise ValueError("latest checkpoint counters differ from the learner")
    if not isinstance(phase, str) or not phase:
        raise ValueError("latest checkpoint phase must be non-empty")
    identity_version = getattr(model, "wbr_run_identity_version", None)
    embedded_identity = getattr(model, "wbr_run_identity_sha256", None)
    if embedded_identity != run_identity_sha256 or not isinstance(
        identity_version, str
    ):
        raise ValueError("latest checkpoint model identity is not bound")
    checkpoint = atomic_save_model(model, output_dir / LATEST_TRAIN_MODEL_FILE)
    state = {
        "schema_version": LATEST_TRAIN_CHECKPOINT_VERSION,
        "file": checkpoint.name,
        "sha256": file_sha256(checkpoint),
        "run_identity_version": identity_version,
        "run_identity_sha256": run_identity_sha256,
        "timesteps": int(timesteps),
        "ppo_updates": int(ppo_updates),
        "phase": phase,
        "role": "latest_standard_ppo_learner_for_verified_resume",
    }
    atomic_write_json(output_dir / LATEST_TRAIN_STATE_FILE, state)
    return state


def save_evaluation_checkpoint(
    model,
    output_dir: Path,
    *,
    file_name: str,
    run_identity_sha256: str,
    role: str,
    metrics: dict[str, float] | None = None,
) -> dict[str, object]:
    """Atomically save one non-resumable diagnostic/selection checkpoint."""

    if Path(file_name).name != file_name or not file_name.endswith(".zip"):
        raise ValueError("evaluation checkpoint file_name must be a local zip name")
    if len(run_identity_sha256) != 64:
        raise ValueError("evaluation checkpoint identity must be a SHA-256 digest")
    if not role:
        raise ValueError("evaluation checkpoint role must be non-empty")
    identity_version = getattr(model, "wbr_run_identity_version", None)
    embedded_identity = getattr(model, "wbr_run_identity_sha256", None)
    if embedded_identity != run_identity_sha256 or not isinstance(
        identity_version, str
    ):
        raise ValueError("evaluation checkpoint model identity is not bound")
    checkpoint = atomic_save_model(model, output_dir / file_name)
    state = _evaluation_state(checkpoint=checkpoint, sha256=file_sha256(checkpoint),
                              identity_version=identity_version, identity=run_identity_sha256,
                              timesteps=int(model.num_timesteps), updates=int(model._n_updates),
                              role=role, metrics=dict(metrics or {}))
    atomic_write_json(checkpoint.with_suffix(".json"), state)
    return state


def validate_latest_train_checkpoint(
    checkpoint: Path,
    *,
    run_identity_version: str,
    run_identity_sha256: str,
) -> dict[str, object]:
    if checkpoint.name != LATEST_TRAIN_MODEL_FILE:
        raise ValueError("latest checkpoint validation requires the canonical file")
    payload = json.loads(
        (checkpoint.parent / LATEST_TRAIN_STATE_FILE).read_text(encoding="utf-8")
    )
    required = {
        "schema_version",
        "file",
        "sha256",
        "run_identity_version",
        "run_identity_sha256",
        "timesteps",
        "ppo_updates",
        "phase",
        "role",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("latest train checkpoint sidecar fields are invalid")
    if payload["schema_version"] != LATEST_TRAIN_CHECKPOINT_VERSION:
        raise ValueError("unsupported latest train checkpoint schema")
    if payload["file"] != checkpoint.name:
        raise ValueError("latest train checkpoint filename mismatch")
    if (
        payload["run_identity_version"] != run_identity_version
        or payload["run_identity_sha256"] != run_identity_sha256
    ):
        raise ValueError("latest train checkpoint identity mismatch")
    if payload["role"] != "latest_standard_ppo_learner_for_verified_resume":
        raise ValueError("latest train checkpoint role is invalid")
    if (
        type(payload["timesteps"]) is not int
        or type(payload["ppo_updates"]) is not int
        or payload["timesteps"] < 0
        or payload["ppo_updates"] < 0
    ):
        raise ValueError("latest train checkpoint counters are invalid")
    if file_sha256(checkpoint) != payload["sha256"]:
        raise ValueError("latest train checkpoint SHA-256 mismatch")
    return payload


__all__ = [
    "LATEST_TRAIN_CHECKPOINT_VERSION",
    "LATEST_TRAIN_MODEL_FILE",
    "LATEST_TRAIN_STATE_FILE",
    "EVALUATION_CHECKPOINT_VERSION",
    "atomic_save_model",
    "save_latest_train_checkpoint",
    "save_evaluation_checkpoint",
    "validate_latest_train_checkpoint",
    "FrozenEvaluationCheckpoint",
    "capture_evaluation_checkpoint",
    "promote_evaluation_checkpoint",
]
