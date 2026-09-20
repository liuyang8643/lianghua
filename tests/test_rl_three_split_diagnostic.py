"""Opt-in diagnostics use synthetic data and a tiny real standard PPO learner."""
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from ai.rl import train as training
from rl_test_data import write_runtime
from test_rl_training_contract import synthetic_financial_snapshot_verifier
from utils.atomic_file import file_sha256


@pytest.mark.parametrize("execution", ["blocking", "overlap"])
def test_three_split_schedule_shared_sha_and_validation_only_selection(
        tmp_path, monkeypatch, synthetic_financial_snapshot_verifier, execution):
    runtime, output = tmp_path / "runtime.npz", tmp_path / "run"
    write_runtime(runtime)
    args = training.build_parser().parse_args([
        "--runtime", str(runtime), "--output", str(output), "--device", "cuda",
        "--train-start", "2020-06-20", "--train-end", "2020-06-24",
        "--validation-start", "2020-06-25", "--validation-end", "2020-06-28",
        "--test-start", "2020-06-29", "--test-end", "2020-07-02",
        "--rollouts", "3", "--eval-every-rollouts", "2", "--n-steps", "4",
        "--n-envs", "1", "--rollout-backend", "dummy", "--backtest-workers", "1",
        "--batch-size", "4", "--n-epochs", "5", "--min-episode-transitions", "1",
        "--advantage-baseline", "none",
        "--evaluation-execution", execution,
        "--learning-rate", "0.0003", "--learning-rate-end-fraction", "0.1",
    ])
    # Stable test-only source identity lets independent source review run in parallel.
    monkeypatch.setattr(training, "policy_source_sha256", lambda _: "d" * 64)
    monkeypatch.setattr(training, "_summary_calmar", lambda _: 1e9)
    fit = Mock(wraps=training._fit_train_normalizer)
    monkeypatch.setattr(training, "_fit_train_normalizer", fit)
    original_prepare = training._prepare_split
    preparation = []

    def prepare(path, start, end, **kwargs):
        split = {args.train_start: "train", args.validation_start: "validation", args.test_start: "test"}[start]
        if split != "train":
            pending = json.loads((output / "evaluation_curves.json").read_text("utf8"))["pending"]
            assert pending["split"] == split
        preparation.append(split)
        return original_prepare(path, start, end, **kwargs)

    monkeypatch.setattr(training, "_prepare_split", prepare)
    original_replay = training._evaluate_frozen_on_descriptor

    def replay(snapshot, descriptor, **kwargs):
        trace = original_replay(snapshot, descriptor, **kwargs)
        returns = np.zeros_like(trace.portfolio_returns)
        if kwargs["task_id"] == "train_candidate":
            returns[:] = -0.001  # No new training best on the final evaluation.
        elif kwargs["task_id"] == "validation_candidate":
            returns[:2] = [-0.01, {0: 0.1, 8: 0.012, 12: 0.02}[snapshot.timesteps]]
        else:
            assert kwargs["task_id"] == "diagnostic_test"
            if snapshot.timesteps:
                selected = json.loads((output / "model.json").read_text("utf8"))
                assert selected["sha256"] == snapshot.sha256
                assert selected["timesteps"] == snapshot.timesteps
                assert selected["role"] == "validation_calmar_selected_checkpoint"
            else:
                assert not (output / "model.zip").exists()
            returns[:2] = [-0.01, 0.2] if snapshot.timesteps == 8 else [-0.03, 0]
        nav = float(trace.nav[0]) * np.r_[1., np.cumprod(1. + returns)]
        return replace(trace, portfolio_returns=returns, nav=nav)

    monkeypatch.setattr(training, "_evaluate_frozen_on_descriptor", replay)
    assert training.train(args) == output.resolve()
    identity = json.loads((output / "run_identity.json").read_text("utf8"))
    assert identity["identity_version"] == training.RUN_IDENTITY_VERSION
    assert identity["contract"]["algorithm"]["evaluation_protocol"] == "periodic_three_split_validation_selection"
    assert identity["contract"]["algorithm"]["n_epochs"] == 5
    assert identity["contract"]["algorithm"]["learning_rate_schedule"]["total_transitions"] == 12
    assert identity["contract"]["evaluation_protocol"]["evaluation_every_rollouts"] == 2
    assert fit.call_count == 1 and preparation == ["train", "validation", "test"]
    curves = json.loads((output / "evaluation_curves.json").read_text("utf8"))
    assert curves["pending"] is None and set(curves["baselines"]) == {"train", "validation", "test"}
    assert set(curves["resident_shared_memory_bytes"]) == {"train", "validation", "test"}
    assert all(size > 0 for size in curves["resident_shared_memory_bytes"].values())
    for split in ("train", "validation", "test"):
        assert [row["timesteps"] for row in curves[split]] == [0, 8, 12]
        assert curves[split][0]["eligible_for_selection"] is False
    for rows in zip(curves["train"], curves["validation"], curves["test"]):
        assert len({row["checkpoint_sha256"] for row in rows}) == 1
    assert curves["train"][-1]["train_improved"] is False
    assert all(not row["eligible_for_selection"] for row in curves["test"])
    metadata = json.loads((output / "training.json").read_text("utf8"))
    assert metadata["status"] == "training_complete"
    assert metadata["checkpoint_selection"]["timesteps"] == 12
    assert curves["test"][1]["test_calmar"] > curves["test"][2]["test_calmar"]
    assert file_sha256(output / "initial_model.zip") == curves["train"][0]["checkpoint_sha256"]
    assert file_sha256(output / "model.zip") == curves["validation"][-1]["checkpoint_sha256"]
    assert json.loads((output / "latest_train_model.json").read_text("utf8"))["timesteps"] == 12
    if execution == "overlap":
        training._validate_evaluation_completion(output, identity)
    assert not (output / "manifest.json").exists()
    assert not (output / "lifecycle_claim.json").exists()
    assert not (output / "pretest_qualification.json").exists()






def test_default_unseeded_root_can_resume_same_contract(tmp_path, synthetic_financial_snapshot_verifier):
    runtime = tmp_path / "runtime.npz"
    write_runtime(runtime)
    args = training.build_parser().parse_args([
        "--runtime", str(runtime), "--output", str(tmp_path / "root"),
        "--train-start", "2020-06-20", "--train-end", "2020-06-24",
        "--validation-start", "2020-06-25", "--validation-end", "2020-06-28",
        "--test-start", "2020-06-29", "--test-end", "2020-07-02",
        "--rollouts", "1", "--n-envs", "1", "--n-steps", "4", "--batch-size", "4",
        "--n-epochs", "1", "--min-episode-transitions", "1", "--lookback", "4",
        "--rollout-backend", "dummy", "--backtest-workers", "1", "--device", "cuda",
        "--advantage-baseline", "none",
    ])
    assert args.seed is None
    parent = training.train(args)
    identity = json.loads((parent / "run_identity.json").read_text("utf8"))
    args.output = str(tmp_path / "child")
    args.resume_from = str(parent / "latest_train_model.zip")
    child = training.train(args)
    continued = json.loads((child / "run_identity.json").read_text("utf8"))
    assert continued["contract"] == identity["contract"]
    curves = json.loads((child / "evaluation_curves.json").read_text("utf8"))
    assert [p["timesteps"] for p in curves["train"]] == [p["timesteps"] for p in curves["validation"]] == [p["timesteps"] for p in curves["test"]]
    assert curves["test"][-1]["timesteps"] == 8
