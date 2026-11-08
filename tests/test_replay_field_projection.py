"""Execution-only projections declare their exact retained source fields."""
from dataclasses import replace

import numpy as np
import pytest

from env.action_schema import ActionSchema
from env.backtest import EXECUTION_RUNTIME_FIELDS, EpisodeSession, prepare_episode_from_runtime, run_day_config_episode
from env.shared_episode import ResidentPreparedEpisode
from factor import precompute_factors
from test_backtest_lightweight import assert_trace_equal, write_canonical_runtime
from test_lightweight_shared_episode import lightweight_episode


@pytest.mark.parametrize('turnover', [0., .2])
@pytest.mark.parametrize('weight', [0., .7])
def test_execution_projection_preserves_detailed_public_replay(lightweight_episode, turnover, weight):
    source = lightweight_episode
    schema = ActionSchema()
    action = np.full(schema.action_dim, weight * 2 - 1)
    action[-1] = turnover / .2 * 2 - 1
    config = schema.decode(action)
    expected = run_day_config_episode(EpisodeSession(source), lambda _: config)
    compact = source.compact_for_replay()
    assert set(compact.runtime.data) == EXECUTION_RUNTIME_FIELDS
    assert compact.compact_for_replay() is compact
    assert compact.runtime.manifest.fields == source.runtime.manifest.fields
    proof = compact.runtime.manifest.replay_projection
    assert proof.source_manifest == source.runtime.manifest
    assert proof.retained_fields == tuple(compact.runtime.data)
    assert proof.schema_version == 'wbr-replay-projection-v2-explicit-fields'
    assert compact.factor_coverage == source.factor_coverage
    assert_trace_equal(expected, run_day_config_episode(EpisodeSession(compact), lambda _: config))
    with ResidentPreparedEpisode(compact) as resident:
        runtime_arrays = dict(resident.descriptor.runtime_data)
        assert set(runtime_arrays) == EXECUTION_RUNTIME_FIELDS
        assert all(not array.flags.writeable for array in resident.episode.runtime.data.values())
        assert_trace_equal(expected, run_day_config_episode(EpisodeSession(resident.episode), lambda _: config))
    with pytest.raises(ValueError, match='reload the original runtime'):
        precompute_factors(compact.runtime)


def test_field_only_projection_at_source_row_zero(tmp_path):
    path = tmp_path / 'runtime.npz'
    write_canonical_runtime(path)
    source = prepare_episode_from_runtime(path, '2020-01-01', '2020-01-04',
                                          prefilter_n=25, encode_observations=False)
    compact = source.compact_for_replay()
    assert compact.runtime.manifest.replay_projection.source_row_offset == 0
    assert compact.runtime.n_dates == source.runtime.n_dates
    assert compact.runtime.manifest.replay_projection.source_manifest == source.runtime.manifest
    assert set(compact.runtime.data) == EXECUTION_RUNTIME_FIELDS
    assert compact.compact_for_replay() is compact


def test_subepisode_projection_keeps_original_provenance(lightweight_episode):
    compact = lightweight_episode.compact_for_replay()
    child = replace(compact, decision_start=compact.decision_start + 1)
    projected = child.compact_for_replay()
    proof = projected.runtime.manifest.replay_projection
    assert proof.source_manifest == lightweight_episode.runtime.manifest
    assert proof.source_row_offset == compact.runtime.manifest.replay_projection.source_row_offset + 1
    assert proof.retained_fields == tuple(projected.runtime.data)
    assert projected.compact_for_replay() is projected
    schema = ActionSchema()
    config = schema.decode(np.zeros(schema.action_dim))
    assert_trace_equal(run_day_config_episode(EpisodeSession(child), lambda _: config),
                       run_day_config_episode(EpisodeSession(projected), lambda _: config))


@pytest.mark.parametrize('change', ['missing', 'extra', 'undeclared_removal'])
def test_projected_materialization_rejects_invalid_field_sets(lightweight_episode, change):
    episode = lightweight_episode.compact_for_replay()
    data = dict(episode.runtime.data)
    proof = episode.runtime.manifest.replay_projection
    if change == 'extra':
        data['volume'] = np.zeros_like(data['open'])
    else:
        data.pop('close')
    manifest = episode.runtime.manifest
    if change == 'undeclared_removal':
        # A self-consistent field declaration still cannot omit execution data.
        proof = replace(proof, retained_fields=tuple(data))
        manifest = replace(manifest, replay_projection=proof)
    with pytest.raises(ValueError, match='retained fields|execution requirements'):
        runtime = replace(episode.runtime, data=data, manifest=manifest)
        replace(episode, runtime=runtime)


@pytest.mark.parametrize('change', ['unknown', 'duplicate', 'legacy_version', 'actor'])
def test_projection_rejects_false_provenance(lightweight_episode, change):
    proof = lightweight_episode.compact_for_replay().runtime.manifest.replay_projection
    changes = {
        'unknown': {'retained_fields': (*proof.retained_fields, 'invented')},
        'duplicate': {'retained_fields': (*proof.retained_fields, proof.retained_fields[0])},
        'legacy_version': {'schema_version': 'wbr-replay-row-projection-v1'},
        'actor': {'observation_schema': 'actor', 'encoded_schema': 'encoder'},
    }
    with pytest.raises(ValueError):
        replace(proof, **changes[change])
