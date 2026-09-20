import numpy as np
import pytest
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import DiagGaussianDistribution
from ai.rl.typed_policy import TypedActorCriticPolicy
from ai.rl.device import load_cuda_ppo
from env.action_schema import ActionSchema
from env.gym_adapter import WBRGymEnv
from rl_test_data import build_episode, fit_normalizer


def _assert_actions(actions, schema):
    for action in actions:
        config = schema.decode(action)
        assert all(0 <= w <= 1 for w in config.factor_weights.values())
        assert 0 <= config.turnover_rate <= .2


def test_gaussian_raw_likelihood_is_not_likelihood_of_clipped_action():
    mean = th.full((4, 12), -3., requires_grad=True)
    log_std = th.zeros(12, requires_grad=True)
    law = DiagGaussianDistribution(12).proba_distribution(mean, log_std)
    raw = mean.detach().clone()
    expected = th.distributions.Normal(mean, log_std.exp()).log_prob(raw).sum(-1)
    th.testing.assert_close(law.log_prob(raw), expected)
    assert not th.equal(law.log_prob(raw), law.log_prob(raw.clamp(-1, 1)))
    config = ActionSchema().decode(raw[0].clamp(-1, 1).numpy())
    assert not any(config.factor_enabled.values())
    assert all(w == 0 for w in config.factor_weights.values())


def test_policy_forward_evaluate_standard_ppo_update_and_reload(tmp_path):
    schema = ActionSchema()
    episode = build_episode(tmp_path / "runtime.npz")
    normalizer = fit_normalizer(episode)
    env = WBRGymEnv(episode, normalizer=normalizer, include_critic_context=True)
    model = PPO(TypedActorCriticPolicy, env, n_steps=8, batch_size=8, n_epochs=1,
                learning_rate=1e-3, policy_kwargs={"action_schema": schema.to_dict(),
                    "encoded_schema": episode.encoder.output_schema.to_dict(), "net_arch": dict(vf=[16])},
                seed=13, device="cuda", verbose=0)
    model.policy.bind_market_store(episode.market_store, normalizer)
    assert isinstance(model.policy.action_dist, DiagGaussianDistribution)
    th.testing.assert_close(model.policy.log_std, th.zeros_like(model.policy.log_std))
    initial, _ = env.reset()
    observations = th.tensor(np.repeat(initial[None], 4, axis=0), device="cuda")
    actions, values, forward_log_prob = model.policy(observations)
    evaluated_values, evaluated_log_prob, entropy = model.policy.evaluate_actions(observations, actions)
    th.testing.assert_close(values, evaluated_values)
    th.testing.assert_close(forward_log_prob, evaluated_log_prob)
    assert entropy is not None and th.isfinite(entropy).all()
    before = model.policy.action_net.weight.detach().clone()
    before_std = model.policy.log_std.detach().clone()
    model.learn(total_timesteps=16)
    assert model.num_timesteps == 16 and model._n_updates == 2
    assert not th.equal(model.policy.action_net.weight.detach(), before)
    assert not th.equal(model.policy.log_std.detach(), before_std)
    _assert_actions(np.clip(model.rollout_buffer.actions.reshape(-1, schema.action_dim), -1, 1), schema)
    assert np.any(np.abs(model.rollout_buffer.actions) > 1)
    expected, _ = model.predict(initial, deterministic=True)
    path = tmp_path / "continuous_policy"
    model.save(path)
    restored = load_cuda_ppo(path, env=env)
    with pytest.raises(RuntimeError, match="bind_market_store"):
        restored.predict(initial, deterministic=True)
    restored.policy.bind_market_store(episode.market_store, normalizer)
    actual, _ = restored.predict(initial, deterministic=True)
    np.testing.assert_array_equal(actual, expected)
    for key, value in model.policy.state_dict().items():
        th.testing.assert_close(restored.policy.state_dict()[key], value, rtol=0, atol=0)


def test_critic_context_is_isolated_from_actor_outputs_and_gradients(tmp_path):
    schema = ActionSchema()
    episode = build_episode(tmp_path / "runtime.npz")
    normalizer = fit_normalizer(episode)
    env = WBRGymEnv(episode, normalizer=normalizer, include_critic_context=True)
    policy = TypedActorCriticPolicy(env.observation_space, env.action_space, lambda _: 1e-3,
        action_schema=schema, encoded_schema=episode.encoder.output_schema.to_dict(),
        net_arch=dict(vf=[16]))
    policy.bind_market_store(episode.market_store, normalizer)
    policy.set_training_mode(False)
    initial, _ = env.reset()
    with pytest.raises(ValueError, match="CUDA"):
        policy.to("cpu")
    with pytest.raises(ValueError, match="CUDA"):
        policy.cpu()
    with pytest.raises(ValueError, match="cuda"):
        policy(th.tensor(initial[None]))
    observation = th.tensor(np.repeat(initial[None], 4, axis=0), device="cuda", requires_grad=True)
    split = policy.actor_observation_dim
    changed = observation.detach().clone()
    changed[:, split:] += 1000
    left = policy.action_net(policy.mlp_extractor.forward_actor(observation))
    right = policy.action_net(policy.mlp_extractor.forward_actor(changed))
    th.testing.assert_close(left, right, rtol=0, atol=0)
    left_action = policy.get_distribution(observation).mode().detach()
    right_action = policy.get_distribution(changed).mode().detach()
    th.testing.assert_close(left_action, right_action, rtol=0, atol=0)
    assert not th.equal(policy.predict_values(observation), policy.predict_values(changed))
    gradient = th.autograd.grad(left[:, :12].sum(), observation)[0]
    assert th.count_nonzero(gradient[:, split:]) == 0
    assert th.count_nonzero(gradient[:, 0]) == 0
    assert th.count_nonzero(gradient[:, 1:split]) > 0


def test_standard_ppo_rejects_cpu_model_placement_during_construction(tmp_path):
    episode = build_episode(tmp_path / "runtime.npz")
    env = WBRGymEnv(episode, normalizer=fit_normalizer(episode), include_critic_context=True)
    with pytest.raises(ValueError, match="CUDA"):
        PPO(TypedActorCriticPolicy, env, n_steps=2, batch_size=2, device="cpu",
            policy_kwargs={"encoded_schema": episode.encoder.output_schema.to_dict()})
