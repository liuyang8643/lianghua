import inspect
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pytest
from stable_baselines3 import PPO
from ai.rl.typed_policy import TypedActorCriticPolicy

from ai.rl.rollout import (
    build_rollout_env,
    build_worker_assignments,
    resolve_rollout_backend,
)
from env.shared_episode import SharedPreparedEpisodeOwner
from env.action_schema import ActionSchema
from rl_test_data import build_episode, fit_normalizer


@pytest.fixture
def episode(tmp_path):
    return build_episode(tmp_path / "runtime.npz")


def test_backend_resolution_and_worker_assignments_are_fold_free():
    assert resolve_rollout_backend("auto", 1) == "dummy"
    assert resolve_rollout_backend("auto", 2) == "subproc"
    assignments = build_worker_assignments(n_envs=4, base_seed=10)

    assert [item.worker_index for item in assignments] == [0, 1, 2, 3]
    assert [item.seed for item in assignments] == [10, 11, 12, 13]
    assert not hasattr(assignments[0], "fold_index")
    assert not hasattr(assignments[0], "region")


def test_rollout_builder_has_no_static_reference_or_window_arguments():
    names = set(inspect.signature(build_rollout_env).parameters)

    assert "reference_action" not in names
    assert "window_transitions" not in names


@pytest.mark.parametrize("backend", ("dummy", "subproc"))
def test_full_period_workers_reset_only_after_all_training_transitions(episode, backend):
    schema = ActionSchema()
    environment = build_rollout_env(
        episode, schema, fit_normalizer(episode), initial_cash=1_000_000.0,
        assignments=build_worker_assignments(n_envs=2, base_seed=17),
        backend=backend, random_window_min_transitions=None,
    )
    try:
        env = environment.vec_env
        env.reset()
        manifest = environment.manifest()
        assert manifest["episode_scope"] == "full_training_period"
        assert manifest["window_start"] == "training_start"
        assert manifest["minimum_window_transitions"] is None
        actions = np.zeros((2, schema.action_dim), dtype=np.float32)
        for _ in range(2):
            for index in range(episode.transition_count):
                _, _, dones, _ = env.step(actions)
                assert dones.tolist() == [index == episode.transition_count - 1] * 2
            assert all(info["episode_transitions"] == episode.transition_count
                       for info in env.reset_infos)
            assert all("random_window" not in info for info in env.reset_infos)
    finally:
        environment.close()


def test_shared_episode_roundtrip_is_read_only_and_contains_runtime_contract(episode):
    owner = SharedPreparedEpisodeOwner.create(episode)
    names = tuple(item.shared_memory_name for item in owner.descriptor.arrays)
    attached = owner.descriptor.attach()
    try:
        assert attached.episode.transition_count == episode.transition_count
        source_rows = np.searchsorted(episode.runtime.trade_dates, attached.episode.runtime.trade_dates)
        np.testing.assert_array_equal(
            attached.episode.runtime.field("listing_age"),
            episode.runtime.field("listing_age")[source_rows],
        )
        np.testing.assert_array_equal(
            attached.episode.runtime.field("delisted_mask"),
            episode.runtime.field("delisted_mask")[source_rows],
        )
        assert attached.episode.runtime.field("open").flags.writeable is False
    finally:
        attached.close()
        owner.close()
    for name in names:
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=name, create=False)


@pytest.mark.parametrize("backend", ("dummy", "subproc"))
def test_every_vector_slot_uses_random_contiguous_train_windows(episode, backend):
    schema = ActionSchema()
    environment = build_rollout_env(
        episode,
        schema,
        fit_normalizer(episode),
        initial_cash=1_000_000.0,
        assignments=build_worker_assignments(n_envs=2, base_seed=17),
        backend=backend,
        random_window_min_transitions=1,
    )
    try:
        observations = environment.vec_env.reset()
        assert observations.shape[0] == 2
        manifest = environment.manifest()
        assert manifest["episode_scope"] == "random_contiguous_train_window"
        assert manifest["minimum_window_transitions"] == 1
        assert manifest["static_reference_in_rollout"] is False
        model = PPO(
            TypedActorCriticPolicy,
            environment.vec_env,
            n_steps=episode.transition_count,
            batch_size=episode.transition_count,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.0,
            n_epochs=1,
            seed=17,
            verbose=0,
            device="cuda",
            policy_kwargs={"encoded_schema": episode.encoder.output_schema.to_dict()},
        )
        model.policy.bind_market_store(episode.market_store, fit_normalizer(episode))
        model.learn(total_timesteps=2 * episode.transition_count)
        assert model.num_timesteps == 2 * episode.transition_count
    finally:
        environment.close()
