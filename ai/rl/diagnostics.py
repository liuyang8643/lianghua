"""Passive PPO diagnostics over the current compact raw-market input contract."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback

from env.metrics import ANNUALIZATION_DAYS
from utils.atomic_file import replace_file, file_sha256
from ai.reporting import append_training_diagnostic


def rollout_gradient_diagnostics(model, last_values, last_dones):
    buffer = model.rollout_buffer
    # Context coordinate 2 is H/(H+252), after the actor-only prefix.
    context_start = model.policy.actor_observation_dim
    saturation = buffer.observations[..., context_start + 2].astype(np.float64)
    horizon = np.rint(ANNUALIZATION_DAYS * saturation / (1.0 - saturation))
    unscaled = copy.copy(buffer)
    unscaled.rewards = (buffer.rewards / (horizon / ANNUALIZATION_DAYS)).astype(np.float32)
    unscaled.advantages = np.empty_like(buffer.advantages)
    unscaled.compute_returns_and_advantage(last_values, last_dones)
    next_values = np.concatenate((buffer.values[1:], last_values.detach().cpu().numpy().reshape(1, -1)))
    next_active = np.concatenate((1.0 - buffer.episode_starts[1:], (1.0 - last_dones).reshape(1, -1)))
    bootstrap = model.gamma * next_values * next_active - buffer.values

    def energy_shares(array):
        square = np.asarray(array, dtype=np.float64) ** 2
        total = float(square.sum())
        return {
            name: float(square[mask].sum() / total) if total else 0.0
            for name, mask in masks.items()
        }

    masks = {'H_le60': horizon <= 60, 'H_61_252': (horizon > 60) & (horizon <= 252),
             'H_253_504': (horizon > 252) & (horizon <= 504), 'H_gt504': horizon > 504}
    # Recreate the next PPO minibatch permutation without advancing its RNG.
    rng_state = np.random.get_state()
    indices = np.random.permutation(buffer.buffer_size * buffer.n_envs)[:model.batch_size]
    np.random.set_state(rng_state)

    def tensor(name):
        values = getattr(buffer, name)
        return th.as_tensor(values[indices % buffer.buffer_size, indices // buffer.buffer_size], device=model.device)

    policy = model.policy
    was_training = policy.training
    policy.set_training_mode(True)
    values, log_prob, _ = policy.evaluate_actions(tensor('observations'), tensor('actions'))
    advantages = tensor('advantages').flatten()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    ratio = th.exp(log_prob - tensor('log_probs').flatten())
    clip_range = model.clip_range(model._current_progress_remaining)
    actor_loss = -th.min(advantages * ratio, advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)).mean()
    critic_loss = model.vf_coef * th.mean((tensor('returns').flatten() - values.flatten()) ** 2)
    parameters = tuple(policy.parameters())
    actor_grads = th.autograd.grad(actor_loss, parameters, allow_unused=True, retain_graph=True)
    critic_grads = th.autograd.grad(critic_loss, parameters, allow_unused=True)

    def norm(grads):
        return float(th.sqrt(sum(th.sum(grad ** 2) for grad in grads if grad is not None)).detach().cpu())

    combined = tuple(a + c if a is not None and c is not None else a if a is not None else c
                     for a, c in zip(actor_grads, critic_grads, strict=True))
    joint_norm = norm(combined)
    names = [name for name, _ in policy.named_parameters()]
    groups = {
        'market_encoder': 'mlp_extractor.raw_features.market_temporal.',
        'history_encoder': 'mlp_extractor.raw_features.history_temporal.',
        'fusion': 'mlp_extractor.raw_features.fusion.',
        'action_head': 'action_net.',
        'exploration': 'log_std',
        'critic_mlp': 'mlp_extractor.value_net.',
    }
    group_diagnostics = {}
    for group, prefix in groups.items():
        selected = [index for index, name in enumerate(names) if name.startswith(prefix)]
        actor_square = sum(float(actor_grads[i].square().sum().detach())
                           for i in selected if actor_grads[i] is not None)
        critic_square = sum(float(critic_grads[i].square().sum().detach())
                            for i in selected if critic_grads[i] is not None)
        dot = sum(float((actor_grads[i] * critic_grads[i]).sum().detach())
                  for i in selected if actor_grads[i] is not None and critic_grads[i] is not None)
        group_diagnostics[f'{group}_actor_norm'] = actor_square ** 0.5
        group_diagnostics[f'{group}_weighted_critic_norm'] = critic_square ** 0.5
        # Undefined cosine is omitted for private actor/critic blocks.
        if actor_square > 0 and critic_square > 0:
            group_diagnostics[f'{group}_gradient_cosine'] = dot / (actor_square * critic_square) ** 0.5
    policy.set_training_mode(was_training)
    result = {
        'timesteps': model.num_timesteps, 'scope': 'first_actual_minibatch_before_optimizer_update',
        'critic_value_rms': float(np.sqrt(np.mean(buffer.values ** 2))),
        'critic_bootstrap_term_rms': float(np.sqrt(np.mean(bootstrap ** 2))),
        'reward_rms': float(np.sqrt(np.mean(buffer.rewards ** 2))),
        'unscaled_reward_rms': float(np.sqrt(np.mean(unscaled.rewards ** 2))),
        'samples': {name: int(mask.sum()) for name, mask in masks.items()},
        'reward_square_shares': energy_shares(buffer.rewards),
        'unscaled_reward_square_shares': energy_shares(unscaled.rewards),
        'advantage_square_shares': energy_shares(buffer.advantages),
        'unscaled_advantage_square_shares': energy_shares(unscaled.advantages),
        'first_minibatch_actor_gradient_norm': norm(actor_grads),
        'first_minibatch_weighted_critic_gradient_norm': norm(critic_grads),
        'first_minibatch_joint_gradient_norm': joint_norm,
        'first_minibatch_clip_multiplier': min(1.0, model.max_grad_norm / (joint_norm + 1e-6)),
        **group_diagnostics,
    }
    return result


class TrainingDiagnostics(BaseCallback):
    """Observe training without drawing actions or changing the optimizer/RNG."""

    def __init__(self, output: Path, *, detailed: bool):
        super().__init__()
        self.output = output
        self.detailed = detailed

    def _on_rollout_start(self) -> None:
        self.execution = {name: [] for name in (
            'gross_turnover_ratio', 'total_cost_ratio', 'total_fees', 'fill_count',
            'portfolio_return', 'exposure', 'annualized_return_increment',
            'drawdown_increment_penalty', 'horizon_scale',
        )}
        self.dates = {}
        self.sample_dates = []
        self.residual_reasons = {}
        self.completed_episodes = 0

    def _on_step(self) -> bool:
        self.completed_episodes += int(np.count_nonzero(self.locals['dones']))
        for info in self.locals['infos']:
            date = str(info['decision_date'])
            self.sample_dates.append(date)
            self.dates[date] = self.dates.get(date, 0) + 1
            reason = str(info['residual_cash_reason'])
            self.residual_reasons[reason] = self.residual_reasons.get(reason, 0) + 1
            for name, values in self.execution.items():
                source = info['episode_reward'] if name in (
                    'annualized_return_increment', 'drawdown_increment_penalty', 'horizon_scale',
                ) else info
                values.append(float(source[name]))
        return True

    def _on_rollout_end(self) -> None:
        started = time.perf_counter()
        model, buffer = self.model, self.model.rollout_buffer
        policy = model.policy
        store = policy.market_store
        raw_input = {
            'schema_version': 'raw-panel-diagnostic-input-v1',
            'compact_schema': dict(policy.encoded_schema),
            'source_observation_schema': store.schema.identifier,
            'normalizer_state_hash': policy.normalizer.state_hash,
            'normalizer_encoder_schema': policy.normalizer.encoder_schema,
            'runtime_source': 'runtime sealed by run_identity_sha256',
            'market_bank': {
                'raw_rows_shape': list(store.raw_rows.shape),
                'raw_rows_dtype': str(store.raw_rows.dtype),
                'pit_universe_mask_shape': list(store.pit_universe_mask.shape),
                'source_row_start': int(store.row_start),
                'decision_start_row': int(store.decision_start),
                'decision_stop_row_exclusive': int(store.decision_stop),
                'decision_date_range': [str(store.decision_dates[0]), str(store.decision_dates[-1])],
            },
            'row_reference_column': 0,
            'row_reference_role': 'routing_only_removed_before_all_learned_layers',
            'observations_payload': 'compact_row_reference_and_dynamic_account_with_critic_context',
            'actor_compact_dimension': int(policy.actor_observation_dim),
            'critic_context_dimension': int(buffer.observations.shape[-1] - policy.actor_observation_dim),
            'raw_market_windows_saved': False,
        }
        self.output.mkdir(parents=True, exist_ok=True)
        arrays = {name: np.asarray(values, dtype=np.float64) for name, values in self.execution.items()}
        quantiles = [0, .01, .5, .99, 1]
        record = {
            'run_identity_sha256': model.wbr_run_identity_sha256, 'timesteps': model.num_timesteps,
            'scope': 'training_rollout_before_update', 'completed_episodes': self.completed_episodes,
            'date_counts': self.dates, 'unique_dates': len(self.dates),
            'residual_cash_reasons': self.residual_reasons,
            'execution': {name: {'mean': float(value.mean()), 'sum': float(value.sum()),
                               'quantiles': np.quantile(value, quantiles).tolist()}
                          for name, value in arrays.items()},
            'quantile_probabilities': quantiles,
            'sampled_action_quantiles': np.quantile(buffer.actions, quantiles, axis=(0, 1)).tolist(),
            'sampled_action_quantiles_scale': 'unclipped_gaussian_box_coordinates',
            'sampled_unit_action_quantiles': np.quantile((np.clip(buffer.actions, -1.0, 1.0) + 1.0) / 2.0, quantiles, axis=(0, 1)).tolist(),
            'clipped_action_fraction': np.mean(np.abs(buffer.actions) > 1.0, axis=(0, 1)).tolist(),
            'action_names': list(policy.typed_action_schema.action_names),
            'raw_input': raw_input,
        }
        if self.detailed:
            count = buffer.buffer_size * buffer.n_envs
            indices = np.linspace(0, count - 1, min(128, count), dtype=int)
            # Time-major sample, matched to execution records; no full observation flatten/copy.
            steps, workers = indices // buffer.n_envs, indices % buffer.n_envs
            observations = buffer.observations[steps, workers]
            was_training = policy.training
            policy.set_training_mode(False)
            try:
                with th.no_grad():
                    features = policy.extract_features(th.as_tensor(observations, device=model.device))
                    latent = policy.mlp_extractor.forward_actor(features)
                    law = policy._get_action_dist_from_latent(latent)
                    gaussian = law.distribution
                    executed_mode = law.mode().clamp(-1.0, 1.0)
                    probe = {
                        'deterministic_actions': executed_mode.cpu().numpy(),
                        'deterministic_unit_actions': ((executed_mode + 1.0) / 2.0).cpu().numpy(),
                        'gaussian_mean': gaussian.mean.cpu().numpy(),
                        'gaussian_std': gaussian.stddev.cpu().numpy(),
                    }
            finally:
                policy.set_training_mode(was_training)
            record['distribution_sample'] = {
                'sample_count': len(indices), 'selection': 'evenly_spaced_time_major_training_rows',
                'sampling_mean': probe['gaussian_mean'].mean(axis=0).tolist(),
                'sampling_std': probe['gaussian_std'].mean(axis=0).tolist(),
                'sampling_scale': '裁剪前高斯Box坐标（可超出-1～1；条件分布均值与标准差）',
                'deterministic_action_mean': probe['deterministic_unit_actions'].mean(axis=0).tolist(),
                'deterministic_action_std': probe['deterministic_unit_actions'].std(axis=0).tolist(),
                'summary_action_scale': 'unit_interval_weights_and_turnover',
            }
            record['gradients'] = rollout_gradient_diagnostics(model, self.locals['values'], self.locals['dones'])
            directory = self.output / 'diagnostic_samples'
            directory.mkdir(exist_ok=True)
            target = directory / f'rollout_{model.num_timesteps:012d}.npz'
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=directory, suffix='.npz', delete=False) as stream:
                    temporary = Path(stream.name)
                    np.savez_compressed(stream, **arrays, **probe, compact_observations=observations,
                                    sample_market_row_references=observations[:, 0].astype(np.int64),
                                    input_metadata_json=np.asarray(json.dumps(raw_input, sort_keys=True)),
                                    sample_time_major_indices=indices,
                                    sample_decision_dates=np.asarray(self.sample_dates)[indices],
                                    decision_dates=np.asarray(self.sample_dates),
                                    run_identity_sha256=np.asarray(model.wbr_run_identity_sha256),
                                    actions=buffer.actions, rewards=buffer.rewards,
                                    advantages=buffer.advantages, returns=buffer.returns,
                                    values=buffer.values, log_probs=buffer.log_probs,
                                        episode_starts=buffer.episode_starts)
                replace_file(temporary, target)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            record['sample_file'] = str(target.relative_to(self.output))
            record['sample_bytes'] = target.stat().st_size
            record['sample_sha256'] = file_sha256(target)
        record['diagnostics_elapsed_seconds'] = time.perf_counter() - started
        scalars = {
            f'execution/{name}': {'label': f'每日 {name} 均值', 'value': values['mean']}
            for name, values in record['execution'].items()
        }
        scalars['unique_dates'] = {'label': '本批不同市场日期数', 'value': record['unique_dates']}
        scalars['completed_episodes'] = {'label': '本批完成账户周期数', 'value': self.completed_episodes}
        scalars['diagnostics_seconds'] = {'label': '本批诊断额外秒数', 'value': record['diagnostics_elapsed_seconds']}
        if self.detailed:
            for name, value in record['gradients'].items():
                if name != 'timesteps' and isinstance(value, (int, float)):
                    scalars[f'gradient/{name}'] = {'label': name, 'value': value}
            for index, name in enumerate(record['action_names']):
                scalars[f'exploration/{name}'] = {'label': f'{name} · 裁剪前高斯标准差',
                    'value': record['distribution_sample']['sampling_std'][index]}
                scalars[f'deterministic/{name}'] = {'label': f'{name} · 确定性动作标准差',
                    'value': record['distribution_sample']['deterministic_action_std'][index]}
        append_training_diagnostic(self.output, algorithm='PPO',
            step=model.num_timesteps / (buffer.buffer_size * buffer.n_envs),
            timesteps=model.num_timesteps, scalars=scalars, details=record)


__all__ = ['TrainingDiagnostics']
