"""Independent actual-history, serialization and journal-chain acceptance."""
from dataclasses import replace
import json
import pickle

import numpy as np
import pytest

from env.action_schema import ActionSchema
from env.backtest import EpisodeSession, prepare_episode_from_runtime
from env.contracts import AccountState, Fill, Observation, OrderPlan, PolicyHistory, PolicyMemory
from env.encoder import ObservationEncoder, RawMarketStore, TrainOnlyNormalizer
from trade.journal import DecisionJournal
from trade.runtime import LiveDecision
from test_backtest_lightweight import write_canonical_runtime
from test_rl_observation import synthetic_inputs, make_builder


def decision(schema, day, previous=PolicyMemory()):
    config = schema.decode(np.zeros(schema.action_dim, dtype=np.float32))
    history = np.zeros((3, schema.action_dim + 3), dtype=np.float32)
    if previous.history is not None:
        count = min(3, len(previous.history.values))
        history[-count:, 0] = 1.0
        history[-count:, 1:] = previous.history.values[-count:]
    observation = Observation(stock_panel=np.zeros((3, 2, 4), dtype=np.float32),
        position_panel=np.zeros((2, 6), dtype=np.float32),
        portfolio=np.zeros(7, dtype=np.float32), policy_history=history,
        time_mask=np.ones(3, dtype=bool), pit_universe_mask=np.ones((3, 2), dtype=bool),
        schema_version="review-observation", decision_date=day)
    account = AccountState(cash=100000.0, nav=100000.0, peak_nav=100000.0)
    plan = OrderPlan(day, (), {"600001.SH": 100}, config,
        {"full_investment_contract_satisfied": True, "prices": {"600001.SH": 10.0}})
    return LiveDecision(observation, schema.encode(config), config, plan, account, previous,
        {"decision_date": day, "runtime_path": "review-local.npz", "runtime_source_sha256": "1" * 64,
         "runtime_schema_hash": "2" * 64, "stock_vocabulary_sha256": "3" * 64, "stock_count": 2,
         "factor_schema_hash": "4" * 64, "observation_schema": observation.schema_version, "prefilter_n": 300},
        {"bundle_version": "review", "created_at": "2026-06-10T00:00:00", "algorithm": "stable_baselines3.PPO",
         "manifest_sha256": "5" * 64, "model_sha256": "6" * 64, "normalizer_sha256": "7" * 64,
         "config_sha256": "8" * 64, "source_sha256": "9" * 64, "action_schema_hash": schema.schema_hash,
         "observation_schema": observation.schema_version, "environment_schema_hash": "a" * 64})


def make_chain(root):
    schema = ActionSchema()
    journal = DecisionJournal(root, schema)
    memory = PolicyMemory()
    for day in ("2026-06-10", "2026-06-11"):
        journal.record_decision(decision(schema, day, memory))
        memory = journal.record_fills(day, (Fill("600001.SH", "buy", 100, 10.0, 1.25, day + "T09:30:00"),))
    return schema, journal, memory


def test_journal_restores_exact_independent_weights_and_actual_ratios(tmp_path):
    schema, journal, memory = make_chain(tmp_path)
    config = schema.decode(np.zeros(schema.action_dim, dtype=np.float32))
    assert memory.previous_day_config == config
    assert memory.previous_gross_turnover_ratio == 0.01
    assert memory.previous_total_cost_ratio == 0.0000125
    replay = journal.replay("2026-06-11")
    assert replay.order_plan.day_config == config
    assert replay.policy_memory_after.previous_day_config == config
    assert replay.policy_memory_after.history.values.tobytes() == memory.history.values.tobytes()
    _, loaded = journal.load_prior_state("2026-06-12")
    assert loaded.history.decision_dates.tobytes() == memory.history.decision_dates.tobytes()
    assert loaded.history.values.tobytes() == memory.history.values.tobytes()


@pytest.mark.parametrize("corruption", ["truncated_history", "old_action", "cost", "date", "version"])
def test_journal_rejects_history_not_derived_from_recorded_decisions_and_fills(tmp_path, corruption):
    _, journal, _ = make_chain(tmp_path)
    path = tmp_path / "2026-06-11" / "fills.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    memory = payload["policy_memory_after"]
    history = memory["history"]
    if corruption == "truncated_history":
        history["decision_dates"] = history["decision_dates"][-1:]
        history["values"] = history["values"][-1:]
    elif corruption == "old_action":
        history["values"][0][0] = 0.25
    elif corruption == "cost":
        memory["previous_total_cost_ratio"] *= 2
        history["values"][-1][-1] = memory["previous_total_cost_ratio"]
    elif corruption == "date":
        history["decision_dates"][-1] = "2026-06-12"
    else:
        payload["journal_version"] = "wbr-live-decision-journal-v3"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)):
        journal.load_prior_state("2026-06-13")
    with pytest.raises((ValueError, RuntimeError)):
        journal.replay("2026-06-11")


def test_every_recorded_action_requires_canonical_float32_coordinates():
    schema = ActionSchema()
    config = schema.decode(np.zeros(schema.action_dim, dtype=np.float32))
    values = np.tile(np.r_[schema.encode(config).astype(np.float64), 0.1, 0.001], (2, 1))
    values[0, schema.action_dim - 2] = 0.123456789
    history = PolicyHistory(np.array(["2026-06-10", "2026-06-11"], dtype="datetime64[D]"), values, schema.schema_hash)
    memory = PolicyMemory(config, 0.1, 0.001, history)
    with pytest.raises(ValueError, match="canonical"):
        schema.validate_policy_memory(memory)


def test_history_owns_immutable_input_buffers():
    schema = ActionSchema()
    config = schema.decode(np.zeros(schema.action_dim, dtype=np.float32))
    dates = np.array(["2026-06-10"], dtype="datetime64[D]")
    values = np.r_[schema.encode(config), 0.1, 0.001].reshape(1, -1)
    history = PolicyHistory(dates, values, schema.schema_hash)
    saved = history.values.tobytes()
    dates[0] = np.datetime64("2030-01-01")
    values[:] = 0.0
    assert str(history.decision_dates[0]) == "2026-06-10"
    assert history.values.tobytes() == saved
    for array in (history.values, history.decision_dates):
        with pytest.raises(ValueError):
            array.flags.writeable = True
    restored = pickle.loads(pickle.dumps(history))
    assert restored == history
    for array in (restored.values, restored.decision_dates):
        with pytest.raises(ValueError):
            array.flags.writeable = True


@pytest.mark.parametrize("corruption", ["action", "plan_date", "missing_previous_fill"])
def test_journal_rejects_inconsistent_serial_decision_records(tmp_path, corruption):
    _, journal, _ = make_chain(tmp_path)
    if corruption == "missing_previous_fill":
        (tmp_path / "2026-06-10" / "fills.json").unlink()
    else:
        path = tmp_path / "2026-06-11" / "decision.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if corruption == "action":
            payload["canonical_action"][0] = 0.25
        else:
            payload["order_plan"]["decision_date"] = "2026-06-12"
        path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        journal.load_prior_state("2026-06-13")
    with pytest.raises((ValueError, FileNotFoundError)):
        journal.replay("2026-06-11")


@pytest.fixture
def long_inputs():
    runtime, factors = synthetic_inputs(date_count=506)
    builder = make_builder(runtime, factors, lookback=504)
    encoder = ObservationEncoder(builder.schema)
    schema = ActionSchema()
    config = schema.decode(np.zeros(schema.action_dim, dtype=np.float32))
    values = np.tile(np.r_[schema.encode(config).astype(np.float64), 0.1, 0.001], (504, 1))
    history = PolicyHistory(runtime.trade_dates[1:505], values, schema.schema_hash)
    memory = PolicyMemory(config, 0.1, 0.001, history)
    account = AccountState(cash=100000.0, nav=100000.0, peak_nav=100000.0)
    return runtime, builder, encoder, schema, config, memory, account


def test_oldest_of_504_actual_records_has_its_own_actor_coordinate(long_inputs):
    runtime, builder, encoder, schema, config, memory, account = long_inputs
    original = builder.build(505, account, policy_memory=memory)
    assert original.stock_panel.shape[0] == original.policy_history.shape[0] == 504
    assert original.policy_history.shape[1] == schema.action_dim + 3
    assert original.policy_history[:, 0].all()
    changed_values = memory.history.values.copy()
    changed_values[0, 0] = 0.5
    changed_memory = replace(memory, history=PolicyHistory(memory.history.decision_dates, changed_values, schema.schema_hash))
    changed = builder.build(505, account, policy_memory=changed_memory)
    store = RawMarketStore.from_observation(original, encoder)
    before, after = encoder.encode(original, store=store), encoder.encode(changed, store=store)
    locations = np.flatnonzero(before != after)
    assert len(locations) == 1
    feature = encoder.output_schema.feature_names[locations[0]]
    assert feature == "policy_history.lag_0504.action.factor_weight.TrueMarketCap"
    assert after[locations[0]] == 0.5
    assert not any(name.startswith("portfolio.previous_") for name in encoder.output_schema.feature_names)


def test_history_lag_one_is_previous_actual_decision_and_calendar_gaps_remain_empty(long_inputs):
    runtime, builder, _, schema, config, _, account = long_inputs
    dates = runtime.trade_dates[[501, 504]]
    values = np.tile(np.r_[schema.encode(config).astype(np.float64), 0.1, 0.001], (2, 1))
    memory = PolicyMemory(config, 0.1, 0.001, PolicyHistory(dates, values, schema.schema_hash))
    history = builder.build(505, account, policy_memory=memory).policy_history
    np.testing.assert_array_equal(np.flatnonzero(history[:, 0]), [500, 503])
    assert not history[501:503].any()
    np.testing.assert_array_equal(history[-1, 1:], values[-1].astype(np.float32))


def test_oldest_market_row_is_kept_in_raw_store_not_compressed(long_inputs):
    _, builder, encoder, _, _, memory, account = long_inputs
    original = builder.build(505, account, policy_memory=memory)
    panel = original.stock_panel.copy()
    feature = builder.schema.stock_feature_names.index("open")
    panel[0, 0, feature] += np.float32(0.5)
    changed = replace(original, stock_panel=panel)
    before = RawMarketStore.from_observation(original, encoder)
    after = RawMarketStore.from_observation(changed, encoder)
    assert after.raw_rows[0, 0, feature] - before.raw_rows[0, 0, feature] == np.float32(.5)
    np.testing.assert_array_equal(encoder.encode(original, store=before), encoder.encode(changed, store=after))


def test_history_scaling_preserves_actual_records_and_padding(long_inputs):
    _, builder, encoder, _, _, memory, account = long_inputs
    obs = builder.build(505, account, policy_memory=memory)
    store = RawMarketStore.from_observation(obs, encoder)
    normalizer = TrainOnlyNormalizer.fit(store, encoder.output_schema, initial_cash=100000., dataset_role="train")
    expected = np.ones(len(encoder.output_schema.history_feature_names), dtype=np.float32)
    expected[-1] = .01
    np.testing.assert_array_equal(normalizer.history_scale, expected)
    cold = builder.build(505, account, policy_memory=PolicyMemory())
    assert not (cold.policy_history / normalizer.history_scale).any()
    np.testing.assert_array_equal(obs.policy_history / normalizer.history_scale,
                                  obs.policy_history / expected)
    for role in ("validation", "test"):
        with pytest.raises(ValueError, match="train"):
            TrainOnlyNormalizer.fit(store, encoder.output_schema, initial_cash=100000., dataset_role=role)


def test_full_episode_504_history_records_actual_fills_and_reset_starts_empty(tmp_path):
    path = tmp_path / "actual-history-runtime.npz"
    write_canonical_runtime(path, stocks=30)
    episode = prepare_episode_from_runtime(path, "2020-06-08", "2020-06-27", lookback=504, prefilter_n=21)
    session = EpisodeSession(episode)
    session.reset()
    schema = session.action_schema
    journal = DecisionJournal(tmp_path / "actual-fill-journal", schema)
    expected = []
    while not session.terminated:
        action = np.full(schema.action_dim, 0.15 if len(expected) % 2 else -0.15, dtype=np.float32)
        config = schema.decode(action)
        before_account = session.current_account
        before_memory = session.current_policy_memory
        before_observation = session.current_observation
        result = session.step(config)
        nav = result.step_result.diagnostics["pretrade_nav"]
        gross = cost = 0.0
        for fill in result.step_result.fills:
            gross += fill.quantity * fill.price
            cost += fill.fee
        turnover = gross / nav
        cost /= nav
        expected.append(np.r_[schema.encode(config).astype(np.float64), turnover, cost])
        np.testing.assert_array_equal(session.current_policy_memory.history.values, expected)
        assert str(session.current_policy_memory.history.decision_dates[-1]) == result.order_plan.decision_date
        observation = session.current_observation
        assert not observation.policy_history[:-len(expected)].any()
        np.testing.assert_array_equal(observation.policy_history[-len(expected):, 1:], np.asarray(expected, dtype=np.float32))
        np.testing.assert_array_equal(episode.encoder.encode(observation, store=episode.market_store), session.encoded_observation)
        day = result.order_plan.decision_date
        template = decision(schema, day, before_memory)
        journal.record_decision(replace(
            template, observation=before_observation,
            action=schema.encode(config), day_config=config, order_plan=result.order_plan,
            account_before=before_account,
            snapshot_identity={**template.snapshot_identity, "observation_schema": before_observation.schema_version},
            policy_identity={**template.policy_identity, "observation_schema": before_observation.schema_version},
        ))
        recorded = journal.record_fills(day, result.step_result.fills)
        assert recorded == session.current_policy_memory
    assert len(expected) == 19
    restored = journal.replay(day).policy_memory_after
    assert restored == session.current_policy_memory
    session.reset(decision_start=episode.decision_start + 7)
    assert session.current_policy_memory.history is None
    assert not session.current_observation.policy_history.any()


def test_history_actor_coordinates_remain_hard_isolated_from_critic_context(long_inputs):
    import gymnasium as gym
    import torch
    from ai.rl.typed_policy import TypedActorCriticPolicy

    torch.set_num_threads(1)
    _, builder, encoder, schema, _, memory, account = long_inputs
    observation = builder.build(505, account, policy_memory=memory)
    store = RawMarketStore.from_observation(observation, encoder)
    normalizer = TrainOnlyNormalizer.fit(store, encoder.output_schema, dataset_role="train", initial_cash=100000.)
    actor = encoder.encode(observation, store=store)
    policy = TypedActorCriticPolicy(
        gym.spaces.Box(-np.inf, np.inf, shape=(len(actor) + 8,), dtype=np.float32),
        gym.spaces.Box(-1.0, 1.0, shape=(schema.action_dim,), dtype=np.float32),
        lambda _: 1e-4, action_schema=schema, encoded_schema=encoder.output_schema.to_dict())
    policy.bind_market_store(store, normalizer)
    policy.set_training_mode(False)
    results = []
    with torch.no_grad():
        for value in (-100.0, 0.0, 100.0):
            inputs = torch.tensor(np.r_[actor, np.full(8, value)][None], dtype=torch.float32, device="cuda")
            law = policy.get_distribution(inputs)
            actions = law.mode().clone()
            results.append((actions, law.log_prob(actions).clone()))
    for actions, log_prob in results[1:]:
        assert torch.equal(actions, results[0][0])
        assert torch.equal(log_prob, results[0][1])
