"""GA, PPO and exact static configs share one declared float32 action precision."""
from dataclasses import replace

import numpy as np
import pytest
import torch as th

from ai.ga import build_individual_config
from ai.ga.config import canonicalize_ga_genes
from ai.policy import FixedPolicy
from ai.rl.evaluation import BacktestRequest, FixedConfigProvider, parallel_backtests_for_episode
from stable_baselines3.common.distributions import DiagGaussianDistribution
from env.action_schema import ActionSchema
from env.backtest import EpisodeSession, run_day_config_episode, run_episode, run_policy_episode
from env.contracts import decode_unit_action, encode_unit_action, replacement_count
from env.gym_adapter import WBRGymEnv
from test_backtest_lightweight import canonical_episode, assert_trace_equal


@pytest.mark.parametrize("buy_n", [50, 300])
def test_all_integer_boundaries_have_identical_static_ga_and_ppo_counts(buy_n):
    schema = ActionSchema(fixed_buy_n=buy_n)
    raw_base = schema.decode(np.zeros(schema.action_dim))
    maximum_count = int(buy_n * schema.layout[-1].maximum)
    parameters = th.full((maximum_count + 1, schema.action_dim), 0.5)
    parameters[:, 11] = th.arange(maximum_count + 1) / maximum_count
    actions = DiagGaussianDistribution(schema.action_dim).proba_distribution(2 * parameters - 1, th.zeros(schema.action_dim)).mode().clamp(-1, 1).numpy()
    for count, action in enumerate(actions):
        literal_rate = count / buy_n
        raw_static = replace(raw_base, turnover_rate=literal_rate)
        ga = schema.canonicalize_day_config(raw_static)
        ppo = schema.decode(action)
        assert raw_static.replacement_limit == count
        assert ga == ppo
        # Field-relative float32 projection precedes the domain's physical-rate
        # quantity codec. Custom portfolio sizes need not retain an unprojected
        # rational boundary, but GA and PPO must use the exact same physical rate.
        physical_thresholds = encode_unit_action(np.arange(1, buy_n + 1) / buy_n)
        expected = int(np.count_nonzero(encode_unit_action(ga.turnover_rate) >= physical_thresholds))
        assert ga.replacement_limit == ppo.replacement_limit == expected
        if buy_n == 50:
            assert expected == count
            generated = build_individual_config(turnover_rate=literal_rate,
                weights=dict.fromkeys(schema.factor_names, 0.5))
            assert schema.from_serialized_day_config(generated) == ppo


@pytest.mark.parametrize("buy_n", [50, 300])
def test_threshold_neighbors_obey_declared_precision_without_epsilon(buy_n):
    thresholds = encode_unit_action(np.arange(1, buy_n + 1, dtype=np.float64) / buy_n)
    for count in range(1, buy_n + 1):
        threshold = thresholds[count - 1]
        center = np.float32(count / buy_n)
        assert replacement_count(buy_n, count / buy_n) == count
        # Arbitrary neighboring Box coordinates may project to the same unit32
        # value. The declared projection, rather than an epsilon, decides this.
        for toward in (-np.inf, np.inf):
            neighbor = np.nextafter(threshold, np.float32(toward))
            if not -1 <= neighbor <= 1:
                continue
            rate = float(decode_unit_action(neighbor))
            projected = encode_unit_action(rate)
            expected = int(np.count_nonzero(projected >= thresholds))
            assert replacement_count(buy_n, rate) == expected
        # The nearest *different canonical* predecessor must remain below.
        predecessor = center
        while encode_unit_action(predecessor) >= threshold:
            predecessor = np.nextafter(predecessor, np.float32(-np.inf))
        assert replacement_count(buy_n, float(predecessor)) == count - 1
        if count < buy_n:
            successor = center
            while encode_unit_action(successor) <= threshold:
                successor = np.nextafter(successor, np.float32(np.inf))
            assert replacement_count(buy_n, float(successor)) == count
            assert replacement_count(buy_n, count / buy_n + 1e-6) == count
        assert replacement_count(buy_n, count / buy_n - 1e-6) == count - 1
    assert replacement_count(buy_n, 0.0) == 0
    assert replacement_count(buy_n, 1.0) == buy_n


def test_ga_gene_canonicalization_is_idempotent_and_never_normalizes_weights():
    schema = ActionSchema()
    genes = build_individual_config(turnover_rate=0.123456789,
        weights={name: (index + 1) / 13 for index, name in enumerate(schema.factor_names)})
    expected = dict(genes)
    for _ in range(100):
        genes, config = canonicalize_ga_genes(genes)
        assert genes == expected
        assert schema.canonicalize_day_config(config) == config
    assert sum(genes["weights"].values()) > 5


def test_real_float32_state_head_and_ga_use_identical_projected_values():
    th.manual_seed(476)
    schema = ActionSchema()
    head = th.nn.Linear(16, schema.action_dim)
    raw = head(th.randn(64, 16)).detach().clamp(-1, 1)
    parameters = (raw + 1) / 2
    actions = raw.numpy()
    for modes, action in zip(parameters.numpy(), actions):
        genes = build_individual_config(turnover_rate=float(modes[-1]) * schema.layout[-1].maximum,
            weights=dict(zip(schema.factor_names, map(float, modes[:-1]))))
        assert schema.from_serialized_day_config(genes) == schema.decode(action)


@pytest.mark.parametrize("rate", [0.0, 0.02, 0.06, 0.1, 0.12, 0.2])
def test_same_canonical_ga_and_ppo_actions_produce_identical_complete_account_paths(canonical_episode, rate):
    schema = ActionSchema()
    config = schema.from_serialized_day_config(build_individual_config(turnover_rate=rate,
        weights=dict.fromkeys(schema.factor_names, 0.5)))
    parameters = th.full((1, schema.action_dim), 0.5)
    parameters[0, schema.action_dim - 1] = rate / schema.layout[-1].maximum
    action = DiagGaussianDistribution(schema.action_dim).proba_distribution(2 * parameters - 1, th.zeros(schema.action_dim)).mode().clamp(-1, 1).numpy()[0]
    ga_trace = run_day_config_episode(EpisodeSession(canonical_episode), lambda _: config)
    ppo_trace = run_episode(WBRGymEnv(canonical_episode), lambda _: action)
    assert_trace_equal(ga_trace, ppo_trace)


def test_static_parser_fixed_providers_ga_and_ppo_share_canonical_execution(canonical_episode):
    schema = ActionSchema()
    payload = schema.to_static_config(schema.decode(np.zeros(schema.action_dim)))
    del payload["factor_enabled"]
    payload["weights"] = {name: (index + 1) * 0.17 for index, name in enumerate(schema.factor_names)}
    payload["turnover_rate"] = 0.12
    config = schema.from_static_config(payload)
    assert config == schema.canonicalize_day_config(config)
    assert config.replacement_limit == 6

    ga = schema.from_serialized_day_config(build_individual_config(
        turnover_rate=payload["turnover_rate"], weights=dict(config.factor_weights)))
    modes = th.tensor([[*config.factor_weights.values(), payload["turnover_rate"] / schema.layout[-1].maximum]])
    parameters = modes
    action = DiagGaussianDistribution(schema.action_dim).proba_distribution(2 * parameters - 1, th.zeros(schema.action_dim)).mode().clamp(-1, 1).numpy()[0]
    assert config == ga == schema.decode(action)

    fixed_trace = run_policy_episode(EpisodeSession(canonical_episode), FixedPolicy(config))
    request = BacktestRequest(
        task_id="static-canonical",
        provider=FixedConfigProvider(schema.to_static_config(config)),
        action_schema_payload=schema.to_dict(), normalizer_payload=None,
        initial_cash=1_000_000.0,
    )
    provider_trace = parallel_backtests_for_episode(
        canonical_episode, (request,), max_workers=1)[request.task_id]
    ga_trace = run_day_config_episode(EpisodeSession(canonical_episode), lambda _: ga)
    ppo_trace = run_episode(WBRGymEnv(canonical_episode), lambda _: action)
    for trace in (provider_trace, ga_trace, ppo_trace):
        assert_trace_equal(fixed_trace, trace)


def test_serialized_day_config_restore_keeps_exact_dto_values():
    schema = ActionSchema()
    config = replace(schema.decode(np.zeros(schema.action_dim)), turnover_rate=0.123456789)
    assert schema.canonicalize_day_config(config).turnover_rate != config.turnover_rate
    assert schema.from_serialized_day_config(schema.to_static_config(config)) == config


@pytest.mark.parametrize("rate", [0.0, 0.2])
def test_all_zero_weights_share_ga_ppo_static_and_account_paths(canonical_episode, rate):
    schema = ActionSchema()
    action = np.full(schema.action_dim, -1.0, dtype=np.float32)
    action[-1] = rate * 10 - 1
    config = schema.decode(action)
    assert not any(config.factor_enabled.values())
    genes = build_individual_config(turnover_rate=rate,
        weights=dict.fromkeys(schema.factor_names, 0.0))
    assert schema.from_serialized_day_config(genes) == config
    payload = schema.to_static_config(config)
    del payload["factor_enabled"]
    assert schema.from_static_config(payload) == config
    assert np.all(canonical_episode.market_at(canonical_episode.decision_start).factor_scores(config) == 0)
    fixed = run_day_config_episode(EpisodeSession(canonical_episode), lambda _: config)
    ppo = run_episode(WBRGymEnv(canonical_episode), lambda _: action)
    assert_trace_equal(fixed, ppo)
    assert np.all(fixed.full_investment_contract)
    assert np.all(fixed.cash >= 0)
