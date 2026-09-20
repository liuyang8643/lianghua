"""Actual dated decisions survive causal padding, reset and action encoding."""
import numpy as np
import pytest

from env.action_schema import ActionSchema
from env.backtest import EpisodeSession, prepare_episode_from_runtime
from env.contracts import PolicyMemory
from env.observation import DEFAULT_LOOKBACK
from test_backtest_lightweight import write_canonical_runtime


def test_history_uses_actual_fills_and_matches_raw_and_cached_actor(tmp_path):
    path = tmp_path / "runtime.npz"
    write_canonical_runtime(path)
    episode = prepare_episode_from_runtime(path, "2020-06-10", "2020-06-24", lookback=3, prefilter_n=25)
    session = EpisodeSession(episode)
    session.reset()
    schema = session.action_schema
    assert not session.current_observation.policy_history.any()
    rng = np.random.default_rng(915)
    recorded = []
    previous_snapshots = []
    for _ in range(6):
        config = schema.decode(rng.uniform(-0.9, 0.9, schema.action_dim))
        transition = session.step(config)
        raw = session.current_observation
        diag = transition.step_result.diagnostics
        recorded.append(np.r_[1.0, schema.encode(config), diag["gross_turnover_ratio"], diag["total_cost_ratio"]].astype(np.float32))
        expected = np.zeros_like(raw.policy_history)
        count = min(3, len(recorded))
        expected[-count:] = recorded[-count:]
        np.testing.assert_array_equal(raw.policy_history, expected)
        np.testing.assert_array_equal(episode.encoder.encode(raw, store=episode.market_store), session.encoded_observation)
        assert str(session.current_policy_memory.history.decision_dates[-1]) == transition.order_plan.decision_date
        previous_snapshots.append((raw.policy_history, raw.policy_history.copy()))
    for values, old in previous_snapshots:
        np.testing.assert_array_equal(values, old)
    with pytest.raises(ValueError):
        session.current_policy_memory.history.values.flags.writeable = True
    session.reset(decision_start=episode.decision_start + 4)
    assert not session.current_observation.policy_history.any()
    assert DEFAULT_LOOKBACK == 64


def test_history_rejects_unknown_past_and_repeated_dates_even_with_one_row():
    schema = ActionSchema()
    config = schema.decode(np.zeros(schema.action_dim))
    settled = PolicyMemory(config, 0.25, 0.0004)
    with pytest.raises(ValueError, match="missing its actual dated history"):
        schema.advance_policy_memory(settled, settled, decision_date="2020-01-02", history_length=3)
    first = schema.advance_policy_memory(PolicyMemory(), settled, decision_date="2020-01-02", history_length=1)
    for day in ("2020-01-02", "2020-01-01"):
        with pytest.raises(ValueError, match="repeat or go backwards"):
            schema.advance_policy_memory(first, settled, decision_date=day, history_length=1)
    with pytest.raises(ValueError, match="exactly once"):
        schema.advance_policy_memory(first, first, decision_date="2020-01-03", history_length=1)


def test_current_or_future_records_cannot_enter_decision_input(tmp_path):
    path = tmp_path / "runtime.npz"
    write_canonical_runtime(path)
    episode = prepare_episode_from_runtime(path, "2020-06-10", "2020-06-24", lookback=3, prefilter_n=25)
    session = EpisodeSession(episode)
    session.reset()
    schema = session.action_schema
    config = schema.decode(np.zeros(schema.action_dim))
    memory = schema.advance_policy_memory(PolicyMemory(), PolicyMemory(config), decision_date="2020-06-10", history_length=3)
    with pytest.raises(ValueError, match="current or future"):
        episode.observation_builder.build_account(episode.decision_start, session.current_account, policy_memory=memory)
