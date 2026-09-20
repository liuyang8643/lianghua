"""Real learner test: run only when no other PPO learner is active."""

import json
import random

import numpy as np
import torch
from stable_baselines3.common.save_util import load_from_zip_file

from ai.rl.train import build_parser, train, _validate_evaluation_completion
from rl_test_data import write_runtime
from test_rl_training_contract import synthetic_financial_snapshot_verifier


def assert_tree_equal(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for item_left, item_right in zip(left, right):
            assert_tree_equal(item_left, item_right)
    else:
        assert left == right


def test_real_blocking_and_overlap_preserve_policy_optimizer_rng_and_train_curve(tmp_path, synthetic_financial_snapshot_verifier):
    runtime = tmp_path / "runtime.npz"
    write_runtime(runtime)
    outputs, rng_states = [], []
    for mode in ("blocking", "overlap"):
        output = tmp_path / mode
        args = build_parser().parse_args([
            "--runtime", str(runtime), "--output", str(output),
            "--train-start", "2020-06-20", "--train-end", "2020-06-24",
            "--validation-start", "2020-06-25", "--validation-end", "2020-06-28",
            "--test-start", "2020-06-29", "--test-end", "2020-07-02",
            "--rollouts", "4", "--n-steps", "4", "--n-envs", "1",
            "--rollout-backend", "dummy", "--backtest-workers", "1",
            "--batch-size", "4", "--n-epochs", "2", "--device", "cuda",
            "--min-episode-transitions", "1", "--verbose", "0",
            "--eval-every-rollouts", "1", "--evaluation-execution", mode,
            "--seed", "2345",
        ])
        assert train(args) == output.resolve()
        outputs.append(output)
        rng_states.append((random.getstate(), np.random.get_state(), torch.get_rng_state().clone()))
        curves = json.loads((output / "evaluation_curves.json").read_text())
        assert all(len(curves[split]) == 5 for split in ("train", "validation", "test"))
        assert list((output / "evaluation_snapshots").iterdir()) == []
    assert_tree_equal(*rng_states)
    for filename in ("initial_model.zip", "latest_train_model.zip", "train_best_model.zip"):
        left = load_from_zip_file(outputs[0] / filename, device="cuda")[1]
        right = load_from_zip_file(outputs[1] / filename, device="cuda")[1]
        assert "policy.optimizer" in left
        assert_tree_equal(left, right)
    curves = [json.loads((path / "training.json").read_text())["training_curve"] for path in outputs]
    for curve in curves:
        assert len(curve) == 5
        for point in curve:
            point.pop("evaluation_elapsed_seconds")
            point.pop("checkpoint_sha256")
            point.pop("run_identity_sha256")
    assert_tree_equal(*curves)
    identity = json.loads((outputs[1] / "run_identity.json").read_text())
    _validate_evaluation_completion(outputs[1], identity)
    # A drained overlap root can continue only from the canonical latest learner.
    child = tmp_path / "continuation"
    args.output = str(child)
    args.resume_from = str(outputs[1] / "latest_train_model.zip")
    args.rollouts = 1
    assert train(args) == child.resolve()
    child_identity = json.loads((child / "run_identity.json").read_text())
    assert child_identity["lineage"]["parent_identity_sha256"] == identity["identity_sha256"]
    assert child_identity["contract"] == identity["contract"]
    _validate_evaluation_completion(child, child_identity)
    latest = json.loads((child / "latest_train_model.json").read_text())
    assert latest["timesteps"] == 20 and latest["ppo_updates"] == 10
    assert not (child / "holdout_state.json").exists()
