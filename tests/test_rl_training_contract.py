import argparse
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from ai.rl import train as train_module
from ai.rl.train import (
    CHECKPOINT_SELECTION_OBJECTIVE,
    LIFECYCLE_CLAIM_FILE,
    RUN_IDENTITY_VERSION,
    RUN_IDENTITY_FILE,
    STATIC_BENCHMARK_FILE,
    TRAIN_BEST_MODEL_FILE,
    INITIAL_MODEL_FILE,
    _training_timesteps_from_rollouts,
    _assert_model_identity,
    _assert_verified_resume_compatible,
    _claim_lifecycle,
    _resolve_batch_size,
    _seal_run_identity,
    _seal_static_benchmark,
    _validate_static_benchmark,
    build_parser,
    train,
)
from ai.bundle import BundleManifest
from ai.rl.device import require_cuda_device
from env.metrics import REWARD_SCHEMA_VERSION
from utils.atomic_file import file_sha256


from rl_test_data import write_runtime


@pytest.mark.parametrize("option", ["--diagnostic-cache-migration", "--action-migration-from"])
def test_removed_migration_cli_is_rejected_before_loading_data(option):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--runtime", "must-not-load.npz", option, "old.json"])


@pytest.fixture
def synthetic_financial_snapshot_verifier(monkeypatch):
    """Only synthetic training integrations bypass the real archive verifier."""
    verifier = Mock(side_effect=lambda runtime_path: {
        "manifest_sha256": "a" * 64,
        # The completion guard re-hashes the runtime file against this value.
        "snapshot_sha256": file_sha256(Path(runtime_path)),
        "financial_identity": {"sha256": "c" * 64},
        "panel_builder_version": "synthetic-financial-panel-v1",
        "financial_replay_version": "synthetic-financial-replay-v1",
        "availability": "synthetic inputs available strictly before decision T",
        "pit_evidence_limit": "unit fixture only; no real-data PIT certification",
    })
    monkeypatch.setattr(train_module, "read_financial_snapshot_manifest", verifier)
    return verifier


@pytest.mark.parametrize("version", [
    "wbr-ppo-run-identity-v56-long-history-actual-actions",
    "wbr-ppo-run-identity-v58-selected11-three-split-diagnostic",
    "wbr-ppo-run-identity-v59-resident-split-diagnostic",
    "wbr-ppo-run-identity-v60-compact-resident-split-diagnostic",
    "wbr-ppo-run-identity-v68-full-collection-diagnostics",
    "wbr-ppo-run-identity-v69-full-collection-diagnostic",
])
def test_previous_run_identity_is_rejected_before_loading_checkpoint(version):
    previous = train_module._seal_run_identity({
        "identity_version": version,
        "contract": {},
    })
    with pytest.raises(ValueError, match="unsupported PPO run identity version"):
        train_module._validate_run_identity(previous)


def test_financial_snapshot_verifier_failure_prevents_split_preparation(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime.npz"
    output = tmp_path / "rejected"
    write_runtime(runtime)
    # Exercise the actual verifier: a synthetic NPZ has no sealed sidecar.
    verifier = Mock(wraps=train_module.read_financial_snapshot_manifest)
    prepare_split = Mock(side_effect=AssertionError("split opened before financial verification"))
    monkeypatch.setattr(train_module, "read_financial_snapshot_manifest", verifier)
    monkeypatch.setattr(train_module, "_prepare_split", prepare_split)
    args = build_parser().parse_args([
        "--runtime", str(runtime), "--output", str(output), "--device", "cuda",
    ])

    with pytest.raises(FileNotFoundError, match=r"runtime\.manifest\.json"):
        train(args)

    verifier.assert_called_once_with(runtime.resolve())
    prepare_split.assert_not_called()
    assert not (output / RUN_IDENTITY_FILE).exists()
    assert not (output / INITIAL_MODEL_FILE).exists()


def _identity(contract, *, parent=None):
    return _seal_run_identity(
        {
            "identity_version": RUN_IDENTITY_VERSION,
            "contract": contract,
            "lineage": {
                "mode": "root" if parent is None else "resume",
                "parent_identity_sha256": parent,
            },
        }
    )


def test_cli_exposes_only_standard_ppo_throughput_controls():
    destinations = {action.dest for action in build_parser()._actions}

    assert "n_steps" in destinations
    assert "train_window_transitions" not in destinations
    assert "gate_recovery" not in destinations
    assert "gamma" not in destinations
    assert "gae_lambda" not in destinations
    assert "eval_every_rollouts" in destinations
    assert "device" in destinations


def test_rollout_budget_is_an_exact_number_of_complete_vector_rollouts():
    assert _training_timesteps_from_rollouts(30, 37_896) == 1_136_880
    assert _training_timesteps_from_rollouts(1, 37_896) == 37_896


def test_only_cuda_device_is_accepted_before_identity(monkeypatch):
    monkeypatch.setattr(train_module.th.cuda, "is_available", lambda: True)
    assert str(require_cuda_device("cuda")) == "cuda"
    for device in ("auto", "cpu"):
        with pytest.raises(ValueError, match="CUDA"):
            require_cuda_device(device)
    monkeypatch.setattr(train_module.th.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        require_cuda_device("cuda")


@pytest.mark.parametrize("rollout_size,expected", [(1280, 640), (320, 320), (64, 64), (257, 257)])
def test_automatic_batch_preserves_complete_minibatches_for_worker_counts(rollout_size, expected):
    batch_size = _resolve_batch_size(0, rollout_size)
    assert batch_size == expected
    assert batch_size > 1 and rollout_size % batch_size == 0


def test_explicit_batch_remains_authoritative_and_must_partition_rollout():
    assert _resolve_batch_size(64, 1280) == 64
    with pytest.raises(ValueError, match="divide"):
        _resolve_batch_size(256, 320)


def test_verified_resume_requires_identical_frozen_contract():
    parent = _identity({"objective": "net_calmar", "n_steps": 3158})
    child = _identity(
        {"objective": "net_calmar", "n_steps": 3158},
        parent=parent["identity_sha256"],
    )
    _assert_verified_resume_compatible(parent, child)

    changed = _identity(
        {"objective": "relative_baseline", "n_steps": 3158},
        parent=parent["identity_sha256"],
    )
    with pytest.raises(ValueError, match="contract"):
        _assert_verified_resume_compatible(parent, changed)


def test_optimizer_settings_are_part_of_the_frozen_contract():
    parent = _identity({"learning_rate": 3e-4, "target_kl": 0.03})
    changed = _identity(
        {"learning_rate": 1e-4, "target_kl": 0.03},
        parent=parent["identity_sha256"],
    )
    with pytest.raises(ValueError, match="contract"):
        _assert_verified_resume_compatible(parent, changed)


def test_parent_model_identity_is_checked_before_rebinding():
    parent = _identity({"objective": "net_calmar", "n_steps": 3158})
    model = SimpleNamespace(
        wbr_run_identity_version=RUN_IDENTITY_VERSION,
        wbr_run_identity_sha256=parent["identity_sha256"],
    )
    _assert_model_identity(model, parent, label="parent selected checkpoint")

    model.wbr_run_identity_sha256 = "f" * 64
    with pytest.raises(ValueError, match="parent selected checkpoint"):
        _assert_model_identity(model, parent, label="parent selected checkpoint")


def test_static_benchmark_cache_is_bound_to_frozen_contract_and_tamper_evident():
    summary = {
        "metrics": {"calmar": 1.25},
        "reward_schema_version": REWARD_SCHEMA_VERSION,
    }
    cache = _seal_static_benchmark(
        contract_sha256="a" * 64,
        train=summary,
        validation=summary,
    )
    validated = _validate_static_benchmark(cache, contract_sha256="a" * 64)
    assert validated["train"] == summary

    tampered = json.loads(json.dumps(cache))
    tampered["train"]["metrics"]["calmar"] = 99.0
    with pytest.raises(ValueError, match="SHA"):
        _validate_static_benchmark(tampered, contract_sha256="a" * 64)


def test_static_benchmark_is_populated_incrementally():
    summary = {
        "metrics": {"calmar": 1.25},
        "reward_schema_version": REWARD_SCHEMA_VERSION,
    }
    cache = _seal_static_benchmark(
        contract_sha256="a" * 64,
        train=summary,
        validation=None,
    )
    validated = _validate_static_benchmark(cache, contract_sha256="a" * 64)
    assert validated["validation"] is None




def test_continuation_claim_cannot_be_reused(tmp_path):
    path = tmp_path / LIFECYCLE_CLAIM_FILE
    _claim_lifecycle(
        path,
        mode="continuation",
        run_identity_sha256="a" * 64,
        child_identity_sha256="b" * 64,
    )

    with pytest.raises(ValueError, match="already"):
        _claim_lifecycle(
            path,
            mode="continuation",
            child_identity_sha256="c" * 64,
            run_identity_sha256="a" * 64,
        )


def test_training_source_declares_the_only_objective_and_standard_ppo():
    source = Path(train_module.__file__).read_text("utf-8")

    assert CHECKPOINT_SELECTION_OBJECTIVE == "full_validation_calmar"
    assert "from stable_baselines3 import PPO" in source
    assert "FoldRobustPPO" not in source
    assert "TrainGateRecovery" not in source
    assert "reference_config=" not in source
    assert "def _json_write" not in source
    assert ".write_text(" not in source
    assert "atomic_write_json(" in source
    assert "PPO_GAMMA = 0.99" in source
    assert "PPO_GAE_LAMBDA = 0.95" in source
    assert '"ent_coef": 0.0' in source
    assert "_pretest_qualification" not in source
    assert "holdout_mode" not in source


def test_parser_requires_runtime_and_keeps_config_as_external_benchmark():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
    args = build_parser().parse_args(["--runtime", "runtime.npz"])

    assert isinstance(args, argparse.Namespace)
    assert args.config == "configs/config.json"
    assert (args.train_start, args.train_end) == ("2004-04-28", "2017-12-31")
    assert (args.validation_start, args.validation_end) == ("2018-01-01", "2022-12-31")
    assert (args.test_start, args.test_end) == ("2023-01-01", "2026-08-28")
    assert args.batch_size == 640
    assert args.rollouts == 100000
    assert args.n_steps == 64
    assert args.n_envs == 20
    assert args.episode_scope == "full"
    assert args.eval_every_rollouts == 50
    assert args.device == "cuda"
    assert args.learning_rate == pytest.approx(1e-3)
    assert args.n_epochs == 3
    assert args.target_kl == pytest.approx(0.03)
    assert args.log_std_init == pytest.approx(-1.6)
    assert args.advantage_baseline == "synchronized_env_row_mean"
    assert args.seed is None
    assert args.learning_rate_end_fraction == 1.0
    assert args.training_slippage_rate == args.evaluation_slippage_rate == pytest.approx(0.0025)




def test_failure_before_report_initialization_keeps_original_error(tmp_path, monkeypatch):
    from types import SimpleNamespace
    def rejected(*_):
        raise RuntimeError('before initialization')
    monkeypatch.setattr(train_module, '_train', rejected)
    with pytest.raises(RuntimeError, match='before initialization'):
        train(SimpleNamespace(output=str(tmp_path / 'absent')))


def test_rejected_output_does_not_overwrite_an_existing_run(tmp_path, monkeypatch):
    from types import SimpleNamespace
    report = tmp_path / 'training_report.json'
    report.write_text('{"state":"complete"}', encoding='utf8')
    def rejected(*_):
        raise FileExistsError('existing output')
    monkeypatch.setattr(train_module, '_train', rejected)
    with pytest.raises(FileExistsError):
        train(SimpleNamespace(output=str(tmp_path)))
    assert report.read_text(encoding='utf8') == '{"state":"complete"}'
