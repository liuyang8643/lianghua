import inspect
import subprocess
import sys

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from env.action_schema import ActionSchema
from env.backtest import (
    CRITIC_CONTEXT_SCALAR_FEATURE_NAMES,
    ENVIRONMENT_SCHEMA_VERSION,
    critic_context_dimension,
    environment_schema_manifest,
)
from env.gym_adapter import WBRGymEnv
from rl_test_data import build_episode, static_config, fit_normalizer


@pytest.fixture
def episode(tmp_path):
    return build_episode(tmp_path / "runtime.npz")


def test_gym_adapter_passes_checker_and_has_one_domain_session(episode):
    env = WBRGymEnv(episode)
    check_env(env, warn=True)

    assert not hasattr(env, "_planner")
    assert not hasattr(env, "_simulator")
    assert env.session._planner is not None
    assert env.session._simulator is not None
    assert env.action_space.shape == (12,)
    assert env.action_space.dtype == np.float32
    low, high = env.action_schema.space_bounds
    np.testing.assert_array_equal(env.action_space.low, low)
    np.testing.assert_array_equal(env.action_space.high, high)


def test_gym_constructor_has_only_the_training_random_window_channel():
    names = set(inspect.signature(WBRGymEnv.__init__).parameters)

    assert "reference_config" not in names
    assert "random_window_min_transitions" in names


def test_episode_always_consumes_all_d_minus_one_transitions(episode):
    schema = ActionSchema()
    env = WBRGymEnv(
        episode,
        action_schema=schema,
        normalizer=fit_normalizer(episode),
    )
    action = schema.encode(static_config(schema))
    _, reset_info = env.reset(seed=5)
    rewards = []
    final_info = None
    while True:
        _, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        final_info = info
        assert truncated is False
        assert "reference_nav" not in info
        assert "gate_aligned_reward" not in info
        assert "episode_reward" in info
        if terminated:
            break

    assert len(rewards) == episode.transition_count
    assert reset_info["episode_transitions"] == episode.transition_count
    metrics = env.session.reward_state.performance_state.performance()
    assert sum(rewards) == pytest.approx(
        episode.transition_count / 252 * (metrics.annualized_return - metrics.max_drawdown)
    )
    assert final_info["full_investment_contract_satisfied"] is True


def test_training_window_randomizes_start_and_length_reproducibly(episode):
    env = WBRGymEnv(episode, random_window_min_transitions=1)
    _, first = env.reset(seed=123)
    _, second = env.reset()
    replay = WBRGymEnv(episode, random_window_min_transitions=1)
    _, replay_first = replay.reset(seed=123)
    _, replay_second = replay.reset()

    assert first == replay_first
    assert second == replay_second
    assert first["random_window"] is True
    assert 1 <= first["episode_transitions"] <= episode.transition_count
    assert (first["decision_date"], first["episode_transitions"]) != (
        second["decision_date"], second["episode_transitions"]
    )


def test_critic_context_is_candidate_only_and_actor_is_public_only(episode):
    env = WBRGymEnv(
        episode,
        normalizer=fit_normalizer(episode),
        include_critic_context=True,
    )
    observation, _ = env.reset(seed=1)

    assert len(CRITIC_CONTEXT_SCALAR_FEATURE_NAMES) == 8
    assert critic_context_dimension(episode.encoder) == 8
    assert observation.shape == (episode.encoder.output_dimension + 8,)
    assert not any("reference" in name for name in CRITIC_CONTEXT_SCALAR_FEATURE_NAMES)


def test_environment_manifest_freezes_dense_incremental_drawdown_semantics(episode):
    manifest = environment_schema_manifest(
        ActionSchema(), episode.encoder, prefilter_n=300
    )

    assert ENVIRONMENT_SCHEMA_VERSION == (
        "wbr-ppo-environment-v37-turnover-floor"
    )
    assert manifest["prefilter"] == {
        "n": 300,
        "source": "previous_day_complete_factor_ranking_plus_holdings",
        "policy_action": False,
    }
    preload = manifest["actor_observation_encoding"]["runtime_preload"]
    assert preload == {
        "formula": (
            "(lookback - 1) + "
            "production_factor_history + decision_lag"
        ),
        "lookback": 64,
        "production_factor_history": 100000,
        "decision_lag": 1,
        "required_rows": 100064,
    }
    transition = manifest["transition"]
    assert transition["objective"] == "annualized_return_minus_max_drawdown"
    assert transition["reward"] == (
        "horizon_scaled_annualized_return_increment_minus_new_max_drawdown_increment"
    )
    assert transition["max_drawdown_penalty_weight"] == pytest.approx(1.0)
    assert transition["episode_sum_identity"] == (
        "H/252*(annualized_net_return_minus_episode_max_drawdown)"
    )
    assert transition["fees_and_slippage"] == "included_once_via_net_nav"
    assert transition["full_investment"] == "planner_contract_fail_closed"


def test_public_star_import_only_exports_the_adapter():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from env.gym_adapter import *; "
                "assert sorted(name for name in globals() if name == 'WBRGymEnv') "
                "== ['WBRGymEnv']"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

