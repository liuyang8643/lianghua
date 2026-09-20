import ast
from pathlib import Path

import numpy as np
import pytest

from env.action_schema import ActionSchema
from env.backtest import (
    EpisodeSession,
    PreparedDecision,
    run_day_config_episode,
    run_episode,
    run_policy_episode,
)
from env.contracts import Observation
from env.gym_adapter import WBRGymEnv
from rl_test_data import build_episode, static_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def episode(tmp_path):
    return build_episode(tmp_path / "runtime.npz")


def _assert_trace_equal(left, right):
    assert left.decision_dates == right.decision_dates
    assert left.next_decision_dates == right.next_decision_dates
    assert left.order_plans == right.order_plans
    assert left.fills == right.fills
    assert left.account_events == right.account_events
    assert left.day_configs == right.day_configs
    np.testing.assert_array_equal(left.rewards, right.rewards)
    np.testing.assert_array_equal(left.portfolio_returns, right.portfolio_returns)
    np.testing.assert_array_equal(left.nav, right.nav)
    np.testing.assert_array_equal(left.actions, right.actions)


def test_gym_and_direct_session_share_the_single_account_timeline(episode):
    schema = ActionSchema()
    action = schema.encode(static_config(schema))
    config = schema.decode(action)
    direct = EpisodeSession(episode, action_schema=schema)
    gym = WBRGymEnv(episode, action_schema=schema)

    direct_observation, direct_info = direct.reset()
    gym_observation, gym_info = gym.reset(seed=11)
    np.testing.assert_array_equal(direct_observation, gym_observation)
    assert direct_info == gym_info
    while not direct.terminated:
        expected = direct.step(config)
        observation, reward, terminated, truncated, info = gym.step(action)
        np.testing.assert_array_equal(expected.observation, observation)
        assert reward == expected.step_result.reward
        assert terminated == expected.step_result.terminated
        assert truncated is False
        assert info["nav"] == expected.info["nav"]
        assert "reference_nav" not in info
        assert direct.current_account == gym.current_account
        assert direct.reward_state == gym.session.reward_state


def test_action_config_and_policy_runners_are_identical(episode):
    schema = ActionSchema()
    action = schema.encode(static_config(schema))
    config = schema.decode(action)
    action_trace = run_episode(
        WBRGymEnv(episode, action_schema=schema),
        lambda _observation: action,
    )
    config_trace = run_day_config_episode(
        EpisodeSession(episode, action_schema=schema),
        lambda _observation: config,
    )

    class FixedPolicy:
        def predict(self, observation: Observation, deterministic: bool = True):
            assert deterministic
            return config

    policy_trace = run_policy_episode(
        EpisodeSession(episode, action_schema=schema),
        FixedPolicy(),
    )
    _assert_trace_equal(action_trace, config_trace)
    _assert_trace_equal(action_trace, policy_trace)
    assert len(action_trace.rewards) == episode.transition_count
    assert action_trace.full_investment_contract_satisfied


@pytest.mark.parametrize("module", ["backtest.py", "action_schema.py"])
def test_canonical_session_and_action_schema_have_no_gym_dependency(module):
    tree = ast.parse((ROOT / "env" / module).read_text("utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert not any(name.startswith("gymnasium") for name in imports)


def test_live_prepared_decision_and_episode_plan_are_identical(episode):
    schema = ActionSchema()
    config = static_config(schema)
    session = EpisodeSession(episode, action_schema=schema)
    session.reset()
    session.step(config)
    live = PreparedDecision.build(
        episode.runtime,
        episode.factors,
        action_schema=schema,
        decision_index=session.current_index,
        lookback=episode.observation_builder.schema.lookback,
        prefilter_n=episode.prefilter_n,
    )

    assert np.array_equal(live.listing_age, episode.runtime.field("listing_age"))
    live_plan = live.plan(
        session.current_account,
        config,
        session.current_policy_memory,
    )
    episode_plan = session.step(config).order_plan
    assert live_plan.sell_orders == episode_plan.sell_orders
    assert live_plan.buy_orders == episode_plan.buy_orders
    assert live_plan.day_config == episode_plan.day_config
    assert (
        live_plan.diagnostics["market_rejection_counts"]
        == episode_plan.diagnostics["market_rejection_counts"]
    )
