import json
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from ai.ga import generate_initial_configs, build_individual_config
from ai.ga.config import canonicalize_ga_genes
from ai.ga.train import ga_optimizer, _select_training_candidate
from ai.ga.comparison import Comparison
from env.action_schema import ActionSchema
from env.backtest import EpisodeSession, PreparedEpisode, run_day_config_episode
from env.shared_episode import SharedPreparedEpisodeOwner
from test_backtest_lightweight import canonical_episode, assert_trace_equal


def test_sampling_and_breeding_preserve_exact_unit_weights():
    random.seed(20260914)
    schema = ActionSchema()
    config = build_individual_config(weights=dict.fromkeys(schema.factor_names, 0.5))
    assert list(canonicalize_ga_genes(config)[1].factor_weights.values()) == [0.5]*11
    initial = generate_initial_configs(64)
    results = [{'individual_config': c, 'calmar': i/10} for i,c in enumerate(initial)]
    population = ga_optimizer(results, {}, population_size=32)
    for c in initial+population:
        canonical, day = canonicalize_ga_genes(c)
        assert canonical == c
        schema.validate_day_config(day)
        assert c['single_buy_pct'] == 1/c['buy_n']
        assert c['rebalance_band_pct'] == schema.fixed_rebalance_band_pct
    assert any(abs(w*10-round(w*10)) > 1e-6 for c in initial for w in c['weights'].values())


def test_nonfinite_candidate_rejected():
    with pytest.raises(ValueError):
        _select_training_candidate({'bad': {'calmar': float('nan'), 'individual_config': generate_initial_configs(1)[0]}})


def test_shared_without_actor_inputs_preserves_complete_execution(canonical_episode):
    full = canonical_episode
    light = PreparedEpisode.build(full.runtime, full.factors, prefilter_n=full.prefilter_n, encode_observations=False)
    schema = ActionSchema()
    config = schema.from_serialized_day_config(build_individual_config(turnover_rate=0.1))
    reference = run_day_config_episode(EpisodeSession(full), lambda _: config)
    with SharedPreparedEpisodeOwner.create(light) as owner:
        with owner.descriptor.attach() as attached:
            assert attached.episode.observation_builder is None
            replay = run_day_config_episode(EpisodeSession(attached.episode), lambda _: config)
            assert_trace_equal(reference, replay)
    # Existing PPO-style shared encoding remains exactly equivalent, too.
    with SharedPreparedEpisodeOwner.create(full) as owner:
        with owner.descriptor.attach() as attached:
            replay = run_day_config_episode(EpisodeSession(attached.episode), lambda _: config)
            assert_trace_equal(reference, replay)


def test_diagnostic_test_cannot_select_or_mutate_ga_candidates(tmp_path):
    c = object.__new__(Comparison)
    c.root = tmp_path
    c.rows = []
    c.selected = None
    c.started = 0
    c.parent_elapsed = 0
    c.args = SimpleNamespace(eval_every_generations=1, generations=2)
    c.publish = lambda: None
    values = iter([{'calmar': 0.7}, {'calmar': -100.0}, {'calmar': 0.6}, {'calmar': 100.0}])
    c.holdout = lambda split, config: next(values)
    candidate = {'individual_config': generate_initial_configs(1)[0], 'metrics': {'calmar': 1.0}}
    original = json.dumps(candidate, sort_keys=True)
    c.evaluate(0, candidate, 1)
    c.evaluate(1, candidate, 2)
    assert c.selected['generation'] == 1
    assert 'test' not in c.selected
    assert json.dumps(candidate, sort_keys=True) == original
