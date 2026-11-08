import json

import numpy as np
import torch as th
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from ai.rl.diagnostics import TrainingDiagnostics
from ai.rl.typed_policy import RAW_PANEL_CONFIG, TypedActorCriticPolicy
from env.action_schema import ActionSchema
from env.gym_adapter import WBRGymEnv
from rl_test_data import build_episode, fit_normalizer


@pytest.mark.parametrize('n_envs', [1, 2])
def test_diagnostic_does_not_change_sampling_buffer_or_optimizer_result(tmp_path, n_envs):
    th.set_num_threads(1)
    episode = build_episode(tmp_path / 'runtime.npz')
    normalizer = fit_normalizer(episode)
    schema = ActionSchema()
    output = tmp_path / 'diagnostics'
    outcomes = []
    for callback in (None, TrainingDiagnostics(output, detailed=True)):
        env = DummyVecEnv([lambda: WBRGymEnv(episode, normalizer=normalizer,
                          include_critic_context=True, random_window_min_transitions=1) for _ in range(n_envs)])
        model = PPO(TypedActorCriticPolicy, env, n_steps=16, batch_size=8,
                    n_epochs=1, seed=123, device='cuda', gamma=.995,
                    policy_kwargs={'action_schema': schema.to_dict(),
                                   'encoded_schema': episode.encoder.output_schema.to_dict(),
                                   'raw_panel_config': dict(RAW_PANEL_CONFIG),
                                   'net_arch': {'pi': [], 'vf': [64, 32]}})
        model.policy.bind_market_store(episode.market_store, normalizer)
        model.wbr_run_identity_sha256 = 'a' * 64
        model.learn(16 * n_envs, callback=callback)
        outcomes.append((
            {key: value.detach().clone() for key, value in model.policy.state_dict().items()},
            {key: getattr(model.rollout_buffer, key).copy()
             for key in ('observations', 'actions', 'rewards', 'advantages', 'returns')},
            th.get_rng_state().clone(), np.random.get_state(),
            [state.clone() for state in th.cuda.get_rng_state_all()],
        ))
        env.close()
    before, after = outcomes
    assert all(th.equal(before[0][key], after[0][key]) for key in before[0])
    assert all(np.array_equal(before[1][key], after[1][key]) for key in before[1])
    assert th.equal(before[2], after[2])
    assert np.array_equal(before[3][1], after[3][1])
    assert before[3][2:] == after[3][2:]
    assert all(th.equal(left, right) for left, right in zip(before[4], after[4], strict=True))
    diagnostic = json.loads((output / 'training_diagnostics.jsonl').read_text().splitlines()[0])['details']['gradients']
    assert sum(diagnostic['samples'].values()) == 16 * n_envs
    assert diagnostic['first_minibatch_joint_gradient_norm'] > 0
    assert 0 < diagnostic['first_minibatch_clip_multiplier'] <= 1

    from utils.atomic_file import file_sha256
    records = [json.loads(line) for line in (output / 'training_diagnostics.jsonl').read_text().splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record['schema_version'] == 'training-diagnostics-v3'
    assert record['algorithm'] == 'PPO'
    details = record['details']
    metadata = details['raw_input']
    assert metadata['compact_schema'] == episode.encoder.output_schema.to_dict()
    assert metadata['source_observation_schema'] == episode.market_store.schema.identifier
    assert metadata['normalizer_state_hash'] == normalizer.state_hash
    assert metadata['normalizer_encoder_schema'] == normalizer.encoder_schema
    assert metadata['row_reference_role'] == 'routing_only_removed_before_all_learned_layers'
    assert metadata['row_reference_column'] == 0
    assert metadata['raw_market_windows_saved'] is False
    assert metadata['market_bank']['raw_rows_shape'] == list(episode.market_store.raw_rows.shape)
    assert metadata['market_bank']['source_row_start'] == episode.market_store.row_start
    assert metadata['market_bank']['decision_start_row'] == episode.market_store.decision_start
    assert metadata['market_bank']['decision_stop_row_exclusive'] == episode.market_store.decision_stop
    assert details['distribution_sample']['sample_count'] == 16 * n_envs
    assert len(details['distribution_sample']['sampling_mean']) == schema.action_dim
    assert 'global_base' not in details['distribution_sample']
    assert 'state_adjustment_mean' not in details['distribution_sample']
    sample_path = output / details['sample_file']
    assert file_sha256(sample_path) == details['sample_sha256']
    with np.load(sample_path, allow_pickle=False) as sample:
        assert 'observations' not in sample.files
        assert not {'raw_rows', 'stock_panel', 'market_windows'} & set(sample.files)
        assert json.loads(sample['input_metadata_json'].item()) == metadata
        compact = sample['compact_observations']
        assert compact.shape == (16 * n_envs, model.observation_space.shape[0])
        assert np.array_equal(sample['sample_market_row_references'], compact[:, 0])
        for reference, date in zip(sample['sample_market_row_references'], sample['sample_decision_dates'], strict=True):
            assert episode.market_store.row_reference(str(date)) == reference
        assert len(sample['decision_dates']) == 16 * n_envs
        assert sample['gaussian_mean'].shape == (16 * n_envs, schema.action_dim)
        assert sample['gaussian_std'].shape == (16 * n_envs, schema.action_dim)
        assert 'global_base' not in sample.files
        assert 'state_adjustments' not in sample.files
        assert sample['run_identity_sha256'].item() == 'a' * 64
        assert np.allclose(sample['deterministic_unit_actions'], (sample['deterministic_actions'] + 1) / 2)
