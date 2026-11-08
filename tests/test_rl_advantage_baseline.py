import argparse

from gymnasium import spaces
import numpy as np
import pytest
import torch as th
from stable_baselines3.common.buffers import RolloutBuffer

from ai.rl.advantage import SynchronizedEnvBaselineRolloutBuffer
from ai.rl import train as training


def _fill(buffer: RolloutBuffer, rng: np.random.Generator, n_steps: int, n_envs: int) -> None:
    for step in range(n_steps):
        buffer.add(
            rng.normal(size=(n_envs, 3)).astype(np.float32),
            rng.normal(size=(n_envs, 2)).astype(np.float32),
            rng.normal(size=n_envs).astype(np.float32),
            np.array([step == 0] * n_envs, dtype=np.float32),
            th.as_tensor(rng.normal(size=(n_envs, 1)).astype(np.float32)),
            th.as_tensor(rng.normal(size=n_envs).astype(np.float32)),
        )


def _pair(n_steps: int, n_envs: int, seed: int) -> tuple[RolloutBuffer, RolloutBuffer, th.Tensor, np.ndarray]:
    observation_space = spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
    action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    plain = RolloutBuffer(n_steps, observation_space, action_space, device="cpu", n_envs=n_envs)
    centered = SynchronizedEnvBaselineRolloutBuffer(
        n_steps, observation_space, action_space, device="cpu", n_envs=n_envs)
    _fill(plain, np.random.default_rng(seed), n_steps, n_envs)
    _fill(centered, np.random.default_rng(seed), n_steps, n_envs)
    rng = np.random.default_rng(seed + 1)
    last_values = th.as_tensor(rng.normal(size=(n_envs, 1)).astype(np.float32))
    dones = np.zeros(n_envs, dtype=np.float32)
    return plain, centered, last_values, dones


def test_row_mean_removed_and_returns_unchanged():
    plain, centered, last_values, dones = _pair(n_steps=7, n_envs=5, seed=3)
    plain.compute_returns_and_advantage(last_values, dones)
    centered.compute_returns_and_advantage(last_values, dones)
    np.testing.assert_allclose(centered.returns, plain.returns)
    np.testing.assert_allclose(centered.values, plain.values)
    expected = plain.advantages - plain.advantages.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(centered.advantages, expected, rtol=0, atol=1e-6)
    np.testing.assert_allclose(centered.advantages.mean(axis=1), 0.0, atol=1e-6)
    assert np.abs(centered.advantages - plain.advantages).max() > 1e-3


def test_single_environment_rejected():
    _, centered, last_values, dones = _pair(n_steps=4, n_envs=1, seed=5)
    with pytest.raises(ValueError):
        centered.compute_returns_and_advantage(last_values, dones)


def _cli(**overrides) -> argparse.Namespace:
    parser = training.build_parser()
    args = parser.parse_args(["--runtime", "unused.npz"])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_cli_baseline_requires_full_scope_and_multiple_envs():
    training._validate_cli(_cli(advantage_baseline="synchronized_env_row_mean", episode_scope="full", n_envs=2))
    with pytest.raises(ValueError):
        training._validate_cli(_cli(advantage_baseline="synchronized_env_row_mean", episode_scope="random"))
    with pytest.raises(ValueError):
        training._validate_cli(_cli(advantage_baseline="synchronized_env_row_mean", n_envs=1))
    with pytest.raises(ValueError):
        training._validate_cli(_cli(log_std_init=float("nan")))


def test_cli_defaults_match_module_constants():
    args = _cli()
    assert args.log_std_init == training.DEFAULT_LOG_STD_INIT
    assert args.advantage_baseline == training.DEFAULT_ADVANTAGE_BASELINE
