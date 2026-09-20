import json
from pathlib import Path

import pytest

from ai.rl.checkpoint import (
    LATEST_TRAIN_MODEL_FILE,
    LATEST_TRAIN_STATE_FILE,
    save_evaluation_checkpoint,
    save_latest_train_checkpoint,
    validate_latest_train_checkpoint,
)


class _Model:
    num_timesteps = 128
    _n_updates = 2
    wbr_run_identity_version = "identity-v1"
    wbr_run_identity_sha256 = "a" * 64

    def save(self, path):
        Path(path).write_bytes(b"standard-ppo")


def test_latest_checkpoint_is_identity_bound_and_validated(tmp_path):
    state = save_latest_train_checkpoint(
        _Model(),
        tmp_path,
        run_identity_sha256="a" * 64,
        timesteps=128,
        ppo_updates=2,
        phase="post_update",
    )

    checkpoint = tmp_path / LATEST_TRAIN_MODEL_FILE
    assert checkpoint.is_file()
    assert state == validate_latest_train_checkpoint(
        checkpoint,
        run_identity_version="identity-v1",
        run_identity_sha256="a" * 64,
    )


def test_latest_checkpoint_rejects_tampered_model(tmp_path):
    save_latest_train_checkpoint(
        _Model(),
        tmp_path,
        run_identity_sha256="a" * 64,
        timesteps=128,
        ppo_updates=2,
        phase="post_update",
    )
    checkpoint = tmp_path / LATEST_TRAIN_MODEL_FILE
    checkpoint.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="SHA-256"):
        validate_latest_train_checkpoint(
            checkpoint,
            run_identity_version="identity-v1",
            run_identity_sha256="a" * 64,
        )


def test_latest_sidecar_has_only_the_verified_resume_contract(tmp_path):
    save_latest_train_checkpoint(
        _Model(),
        tmp_path,
        run_identity_sha256="a" * 64,
        timesteps=128,
        ppo_updates=2,
        phase="post_update",
    )
    payload = json.loads((tmp_path / LATEST_TRAIN_STATE_FILE).read_text("utf-8"))

    assert set(payload) == {
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
    assert payload["role"] == "latest_standard_ppo_learner_for_verified_resume"


def test_evaluation_checkpoint_is_identity_bound_with_metric_sidecar(tmp_path):
    state = save_evaluation_checkpoint(
        _Model(),
        tmp_path,
        file_name="train_best_model.zip",
        run_identity_sha256="a" * 64,
        role="best_complete_training_calmar_diagnostic_only",
        metrics={"train_calmar": 1.25},
    )

    assert state["metrics"] == {"train_calmar": 1.25}
    assert (tmp_path / "train_best_model.zip").is_file()
    payload = json.loads((tmp_path / "train_best_model.json").read_text("utf-8"))
    assert payload["sha256"] == state["sha256"]
    assert payload["role"] == "best_complete_training_calmar_diagnostic_only"
