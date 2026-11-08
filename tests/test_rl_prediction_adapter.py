"""Cold live snapshots and frozen offline replay resolve the same raw input."""
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from ai.rl.evaluation import _FrozenPPOCallable
from ai.rl.policy import RLPolicy, predict_encoded_action
from ai.rl.typed_policy import TypedActorCriticPolicy
from env.action_schema import ActionSchema
from env.gym_adapter import WBRGymEnv
from rl_test_data import build_episode, fit_normalizer


def test_live_and_frozen_adapters_have_identical_raw_input_and_output(tmp_path):
    torch.set_num_threads(1)
    episode = build_episode(tmp_path / "runtime.npz")
    schema = ActionSchema()
    normalizer = fit_normalizer(episode)
    env = WBRGymEnv(episode, normalizer=normalizer, include_critic_context=True)
    compact, _ = env.reset()
    model = PPO(TypedActorCriticPolicy, env, n_steps=2, batch_size=2, n_epochs=1,
                seed=17, device="cuda",
                policy_kwargs={"encoded_schema": episode.encoder.output_schema.to_dict()})
    live = RLPolicy(model, schema, episode.encoder, normalizer, prefilter_n=300)
    live_action = live._predict_action(env.session.current_observation)
    live_reference = model.policy.market_store.row_reference(env.session.current_observation.decision_date)
    model.policy.bind_market_store(episode.market_store, normalizer)
    frozen_action = _FrozenPPOCallable(model, schema)(compact)
    np.testing.assert_array_equal(live_action, frozen_action)
    assert live_reference == episode.observation_builder.lookback - 1
    public = compact[:episode.encoder.output_dimension]
    np.testing.assert_array_equal(predict_encoded_action(model, public), frozen_action)
    with pytest.raises(ValueError, match="dimensions"):
        predict_encoded_action(model, np.zeros(4, dtype=np.float32))
    env.close()
