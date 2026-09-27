import json
from pathlib import Path

import numpy as np
import pytest

from ai.rl.train import build_parser
from env.action_schema import ActionSchema
from env.backtest import prepare_episode_from_runtime
from factor.registry import (
    STATIC4_FACTOR_NAMES, STATIC4_FACTORS, STATIC4_SCHEMA_VERSION,
    PRODUCTION_FACTOR_NAMES, policy_factor_definitions,
)
from factor_db.factors.AmihudIlliquidity import AmihudIlliquidity
from rl_test_data import write_runtime, fit_normalizer
from ai.rl.rollout import build_rollout_env, build_worker_assignments


def test_static4_uses_original_factor_and_explicit_new_identity(tmp_path):
    runtime = tmp_path / 'runtime.npz'
    write_runtime(runtime)
    schema = ActionSchema(factor_names=STATIC4_FACTOR_NAMES,
                          schema_version='day-config-static4-v2-live-validity-hold20',
                          fixed_filter_flags=(False, False))
    episode = prepare_episode_from_runtime(runtime, '2020-06-01', '2020-07-15',
                                           lookback=8, prefilter_n=300, action_schema=schema)
    assert episode.factors.schema_version == STATIC4_SCHEMA_VERSION
    assert episode.factors.factor_names == STATIC4_FACTOR_NAMES
    assert episode.encoder.output_schema is not None
    assert schema.action_dim == 5
    assert schema.schema_hash != ActionSchema().schema_hash
    assert issubclass(STATIC4_FACTORS[-1].implementation, AmihudIlliquidity)
    config = json.loads((Path(__file__).parents[1] / 'configs/ppo_static4.json').read_text())
    static = schema.from_static_config(config)
    np.testing.assert_allclose(list(static.factor_weights.values()), [.45, .05, .30, .20], atol=1e-7)
    assert not any(static.filter_flags.values())
    workers = build_rollout_env(
        episode, schema, fit_normalizer(episode), initial_cash=1_000_000,
        assignments=build_worker_assignments(n_envs=2, base_seed=26),
        backend='subproc', random_window_min_transitions=None,
    )
    try:
        workers.vec_env.reset()
        observations, rewards, _, _ = workers.vec_env.step(np.zeros((2, 5), dtype=np.float32))
        assert observations.shape[0] == 2
        assert np.isfinite(rewards).all()
    finally:
        workers.close()


def test_unknown_vocabulary_cannot_enter_encoded_policy():
    with pytest.raises(ValueError, match='unregistered'):
        policy_factor_definitions(('AmihudIlliquidity',))
    assert len(policy_factor_definitions(PRODUCTION_FACTOR_NAMES)) == 12


def test_static4_profile_is_opt_in():
    assert build_parser().parse_args(['--runtime', 'x']).factor_profile == 'production12'
    assert build_parser().parse_args(['--runtime', 'x', '--factor-profile', 'static4']).factor_profile == 'static4'


def test_live_embedded_gates_apply_before_factor_ranking():
    rows, columns = 90, 4
    panel = {name: np.full((rows, columns), value, dtype=np.float64)
             for name, value in [('open', 10), ('close', 10), ('total_share', 1e8),
                                 ('volume', 1e6), ('amount', 1e7)]}
    panel['st_mask'] = np.zeros((rows, columns), dtype=bool)
    panel['st_mask'][:, 0] = True
    panel['open'][:, 1] = 1.99
    panel['open'][:, 2] = 2.0
    for definition in STATIC4_FACTORS:
        values = definition.implementation().calc_batch(panel)
        assert np.isnan(values[-1, :2]).all()
        assert np.isfinite(values[-1, 2:]).all()
