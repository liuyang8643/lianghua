"""The CUDA graph replays the existing frozen actor and expires with its state."""
import numpy as np
import pytest
import torch as th

from ai.rl.typed_policy import TypedActorCriticPolicy
from env.gym_adapter import WBRGymEnv
from rl_test_data import build_episode, fit_normalizer


@pytest.fixture
def frozen_policy(tmp_path):
    th.set_num_threads(1)
    episode = build_episode(tmp_path / "runtime.npz")
    normalizer = fit_normalizer(episode)
    env = WBRGymEnv(episode, normalizer=normalizer, include_critic_context=True)
    policy = TypedActorCriticPolicy(env.observation_space, env.action_space, lambda _: 1e-3,
                                   encoded_schema=episode.encoder.output_schema.to_dict())
    policy.bind_market_store(episode.market_store, normalizer)
    policy.set_training_mode(False)
    observation, _ = env.reset()
    yield policy, th.tensor(observation[None], device="cuda"), episode, normalizer
    env.close()


@th.no_grad()
def eager_action(policy, observation):
    return policy.get_distribution(observation).mode().clone()


def test_graph_reuses_the_actor_with_changing_dates_account_and_history(frozen_policy):
    policy, observation, episode, _ = frozen_policy
    raw = policy.mlp_extractor.raw_features
    graph = None
    rng = th.cuda.get_rng_state().clone()
    for offset in range(3):
        observation[:, 0] = episode.market_store.decision_start + offset
        observation[:, raw.position_stop] = 100_000 + offset * 10_000
        history = observation[:, raw.portfolio_stop:raw.dimension].reshape(
            1, raw.lookback, raw.history_features)
        if offset:
            history[:, -offset:, 0] = 1
            history[:, -offset:, 1:] = 0.3
        expected = eager_action(policy, observation)
        actual = policy._predict(observation, deterministic=True)
        th.testing.assert_close(actual, expected, rtol=0, atol=0)
        if graph is not None:
            assert policy._frozen_actor_graph is graph
        graph = policy._frozen_actor_graph
        changed_context = observation.clone()
        changed_context[:, policy.actor_observation_dim:] += 1000
        th.testing.assert_close(policy._predict(changed_context, deterministic=True), actual, rtol=0, atol=0)
    assert th.equal(rng, th.cuda.get_rng_state())


def test_optimizer_changes_expire_graph_and_market_cache_at_their_own_scope(frozen_policy):
    policy, observation, _, _ = frozen_policy
    raw = policy.mlp_extractor.raw_features
    policy._predict(observation, deterministic=True)
    graph = policy._frozen_actor_graph
    market = raw.prepare_frozen_market()
    for parameter in (policy.action_net.bias, raw.market_temporal.projection.bias):
        policy.optimizer.zero_grad()
        parameter.sum().backward()
        policy.optimizer.step()
        actual = policy._predict(observation, deterministic=True)
        assert policy._frozen_actor_graph is not graph
        graph = policy._frozen_actor_graph
        if parameter is policy.action_net.bias:
            assert raw.prepare_frozen_market() is market
        else:
            assert raw.prepare_frozen_market() is not market
        th.testing.assert_close(actual, eager_action(policy, observation), rtol=0, atol=0)


def test_shape_rebind_and_dtype_changes_rebuild_the_graph(frozen_policy):
    policy, observation, episode, normalizer = frozen_policy
    policy._predict(observation, deterministic=True)
    first = policy._frozen_actor_graph
    batch = observation.repeat(2, 1)
    action = policy._predict(batch, deterministic=True)
    assert policy._frozen_actor_graph is not first
    th.testing.assert_close(action, eager_action(policy, batch), rtol=0, atol=0)
    policy.bind_market_store(episode.market_store, normalizer)
    assert policy._frozen_actor_graph is None
    policy._predict(observation, deterministic=True)
    previous = policy._frozen_actor_graph
    policy.to(dtype=th.float64)
    actual = policy._predict(observation.double(), deterministic=True)
    assert policy._frozen_actor_graph is not previous
    th.testing.assert_close(actual, eager_action(policy, observation.double()), rtol=0, atol=0)


@pytest.mark.parametrize("invalid", [float("nan"), 1.5, -1, 1_000_000])
def test_invalid_references_are_rejected_before_replaying_a_cached_graph(frozen_policy, invalid):
    policy, observation, _, _ = frozen_policy
    policy._predict(observation, deterministic=True)
    observation[:, 0] = invalid
    with pytest.raises(ValueError, match="row reference"):
        policy._predict(observation, deterministic=True)


def test_graph_is_transient_and_checkpoint_reload_requires_binding(frozen_policy, tmp_path):
    policy, observation, episode, normalizer = frozen_policy
    expected = policy._predict(observation, deterministic=True)
    assert not any("frozen" in name or "bank" in name for name in policy.state_dict())
    path = tmp_path / "graph_policy.pth"
    policy.save(path)
    loaded = TypedActorCriticPolicy.load(path)
    loaded.set_training_mode(False)
    assert loaded._frozen_actor_graph is None and loaded.market_store is None
    with pytest.raises(RuntimeError, match="bind_market_store"):
        loaded._predict(observation, deterministic=True)
    loaded.bind_market_store(episode.market_store, normalizer)
    actual = loaded._predict(observation, deterministic=True)
    np.testing.assert_array_equal(actual.cpu().numpy(), expected.cpu().numpy())


def test_sampling_and_differentiable_training_do_not_use_the_graph(frozen_policy):
    policy, observation, _, _ = frozen_policy
    policy._predict(observation, deterministic=False)
    assert policy._frozen_actor_graph is None
    policy._predict(observation, deterministic=True)
    policy.set_training_mode(True)
    assert policy._frozen_actor_graph is None
    actions, _, _ = policy(observation)
    values, likelihood, _ = policy.evaluate_actions(observation, actions.detach())
    assert values.requires_grad and likelihood.requires_grad
    policy.optimizer.zero_grad()
    (values.square().mean() - likelihood.mean()).backward()
    assert th.isfinite(policy.action_net.weight.grad).all()
    assert policy.action_net.weight.grad.abs().sum() > 0
    policy.optimizer.step()
    assert policy._frozen_actor_graph is None
    policy.set_training_mode(False)
    th.testing.assert_close(policy._predict(observation, deterministic=True),
                            eager_action(policy, observation), rtol=0, atol=0)


@th.no_grad()
def test_frozen_predictions_own_their_outputs_across_replays(frozen_policy):
    policy, observation, episode, _ = frozen_policy
    first = policy._predict(observation, deterministic=True)
    saved = first.clone()
    graph = policy._frozen_actor_graph
    assert first.data_ptr() != graph.outputs[0].data_ptr()
    changed = observation.clone()
    changed[:, 0] = episode.market_store.decision_start + 3
    changed[:, policy.mlp_extractor.raw_features.position_stop] += 10_000
    second = policy._predict(changed, deterministic=True)
    assert first.data_ptr() != second.data_ptr()
    assert policy._frozen_actor_graph is graph
    th.testing.assert_close(first, saved, rtol=0, atol=0)
    th.testing.assert_close(second, eager_action(policy, changed), rtol=0, atol=0)


@th.no_grad()
def test_mixed_prediction_and_distribution_calls_refresh_the_current_law(frozen_policy):
    policy, observation, episode, _ = frozen_policy
    second = observation.clone()
    second[:, 0] = episode.market_store.decision_start + 3
    second[:, policy.mlp_extractor.raw_features.position_stop] += 10_000
    rng = th.cuda.get_rng_state().clone()
    policy._predict(observation, deterministic=True)
    assert th.equal(rng, th.cuda.get_rng_state())
    actual_sample = policy._predict(second, deterministic=False)
    sample_rng = th.cuda.get_rng_state().clone()
    th.cuda.set_rng_state(rng)
    distribution = policy.get_distribution(second)
    expected_sample = distribution.sample()
    th.testing.assert_close(actual_sample, expected_sample, rtol=0, atol=0)
    assert th.equal(sample_rng, th.cuda.get_rng_state())
    expected_mode = distribution.mode().clone()
    th.testing.assert_close(policy._predict(second, deterministic=True), expected_mode, rtol=0, atol=0)
    policy._predict(observation, deterministic=True)
    _, actual_log, actual_entropy = policy.evaluate_actions(second, actual_sample)
    distribution = policy.get_distribution(second)
    th.testing.assert_close(actual_log, distribution.log_prob(actual_sample), rtol=0, atol=0)
    th.testing.assert_close(actual_entropy, distribution.entropy(), rtol=0, atol=0)


@th.no_grad()
def test_frozen_mode_rejects_nonfinite_mean(frozen_policy):
    policy, observation, _, _ = frozen_policy
    policy._predict(observation, deterministic=True)
    previous = policy._frozen_actor_graph
    policy.action_net.bias.add_(.1)
    th.testing.assert_close(policy._predict(observation, deterministic=True),
                            eager_action(policy, observation), rtol=0, atol=0)
    assert policy._frozen_actor_graph is not previous
    policy.action_net.bias.fill_(float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        policy._predict(observation, deterministic=True)


@th.no_grad()
def test_public_predict_clips_gaussian_mean_before_domain_decode(frozen_policy):
    policy, observation, _, _ = frozen_policy
    policy.action_net.weight.zero_()
    policy.action_net.bias.fill_(-5.)
    policy.action_net.bias[-1] = 5.
    raw = policy._predict(observation, deterministic=True)
    assert raw[0, 0] == -5 and raw[0, -1] == 5
    executed, _ = policy.predict(observation.cpu().numpy(), deterministic=True)
    np.testing.assert_array_equal(executed, raw.clamp(-1, 1).cpu().numpy())
    config = policy.typed_action_schema.decode(executed[0])
    assert not any(config.factor_enabled.values())
    assert config.turnover_rate == .2


@pytest.mark.parametrize("invalid", [float("nan"), -1000.])
@th.no_grad()
def test_frozen_and_eager_reject_invalid_gaussian_scale(frozen_policy, invalid):
    policy, observation, _, _ = frozen_policy
    policy._predict(observation, deterministic=True)
    policy.log_std.fill_(invalid)
    with pytest.raises(ValueError):
        policy.get_distribution(observation)
    with pytest.raises(ValueError, match="Gaussian scale"):
        policy._predict(observation, deterministic=True)
