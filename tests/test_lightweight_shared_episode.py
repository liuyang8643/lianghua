"""GA and PPO attach the same immutable episode; only PPO needs actor caches."""
from dataclasses import replace
from multiprocessing.shared_memory import SharedMemory
import gc
import weakref

import numpy as np
import pytest

from env.action_schema import ActionSchema
from env.backtest import EpisodeSession, prepare_episode_from_runtime, run_day_config_episode
import env.shared_episode as shared_episode
from env.shared_episode import SharedPreparedEpisodeOwner, ResidentPreparedEpisode
from test_backtest_lightweight import write_canonical_runtime, assert_trace_equal


def test_resident_releases_preparation_and_unlinks_after_exception(tmp_path):
    path = tmp_path / 'runtime.npz'
    write_canonical_runtime(path)
    prepared = prepare_episode_from_runtime(path, '2020-06-20', '2020-06-24',
                                           prefilter_n=25, encode_observations=False)
    source = weakref.ref(prepared.factors.raw)
    schema = ActionSchema()
    config = schema.decode(np.zeros(schema.action_dim))
    expected = run_day_config_episode(EpisodeSession(prepared), lambda _: config)
    with pytest.raises(RuntimeError, match='evaluation failure'):
        with ResidentPreparedEpisode(prepared) as resident:
            names = [array.shared_memory_name for array in resident.descriptor.arrays]
            del prepared
            gc.collect()
            assert source() is None
            assert resident.episode is resident.episode
            assert not resident.episode.factors.raw.flags.writeable
            actual = run_day_config_episode(EpisodeSession(resident.episode), lambda _: config)
            assert_trace_equal(expected, actual)
            raise RuntimeError('evaluation failure')
    resident.close()
    with pytest.raises(RuntimeError, match='closed'):
        resident.episode
    for name in names:
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=name, create=False)


def test_resident_descriptor_does_not_reconstruct_coordinator(lightweight_episode, monkeypatch):
    def unexpected_attach(*_):
        pytest.fail('descriptor-only consumer must not reconstruct coordinator markets')
    monkeypatch.setattr(shared_episode.SharedPreparedEpisodeDescriptor, 'attach', unexpected_attach)
    with ResidentPreparedEpisode(lightweight_episode) as resident:
        assert resident.descriptor.shared_memory_bytes > 0


@pytest.mark.parametrize('encode', [False, True])
def test_shared_episode_with_and_without_actor_cache(tmp_path, encode):
    path = tmp_path / 'runtime.npz'
    write_canonical_runtime(path)
    episode = prepare_episode_from_runtime(path, '2020-06-20', '2020-06-24',
                                          prefilter_n=25, encode_observations=encode)
    schema = ActionSchema()
    config = schema.decode(np.zeros(schema.action_dim))
    expected = run_day_config_episode(EpisodeSession(episode), lambda _: config)
    with SharedPreparedEpisodeOwner.create(episode) as owner:
        descriptor = owner.descriptor
        assert descriptor.listing_age is dict(descriptor.runtime_data)['listing_age']
        names = [item.shared_memory_name for item in descriptor.arrays]
        assert len(names) == len(set(names))
        assert descriptor.shared_memory_bytes == sum(item.nbytes for item in descriptor.arrays)
        with owner.descriptor.attach() as attached:
            other = attached.episode
            assert other.listing_age is other.runtime.field('listing_age')
            assert not other.listing_age.flags.writeable
            assert (other.observation_builder is not None) is encode
            assert (other.market_store is not None) is encode
            for index, name in enumerate(other.factors.factor_names):
                row = other.market_at(other.decision_start).factor_ranks[name]
                assert np.shares_memory(row, other.factors.ranks)
                assert not row.flags.writeable
            for field in ('open', 'preClose'):
                row = getattr(other.market_at(other.decision_start),
                              'open_prices' if field == 'open' else 'preclose_prices')
                assert np.shares_memory(row, other.runtime.field(field))
                assert not row.flags.writeable
            actual = run_day_config_episode(EpisodeSession(other), lambda _: config)
            assert_trace_equal(expected, actual)
        assert attached.closed
    for name in names:
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=name, create=False)


@pytest.fixture
def lightweight_episode(tmp_path):
    path = tmp_path / 'runtime.npz'
    write_canonical_runtime(path)
    return prepare_episode_from_runtime(
        path, '2020-06-20', '2020-06-24', prefilter_n=25,
        encode_observations=False,
    )


@pytest.mark.parametrize('source', ['same_view', 'equal_copy', 'changed_copy', 'reversed_view'])
def test_listing_age_alias_requires_the_exact_readonly_view(lightweight_episode, source):
    episode = lightweight_episode
    original = episode.listing_age
    if source == 'same_view':
        ages = original.view()
    elif source == 'reversed_view':
        ages = original[:, ::-1]
    else:
        ages = original.copy()
        if source == 'changed_copy':
            ages[:, 0] = -1
        ages.flags.writeable = False
    episode = replace(episode, listing_age=ages)
    schema = ActionSchema()
    config = schema.decode(np.zeros(schema.action_dim))
    expected = run_day_config_episode(EpisodeSession(episode), lambda _: config)
    with SharedPreparedEpisodeOwner.create(episode) as owner:
        descriptor = owner.descriptor
        runtime_age = dict(descriptor.runtime_data)['listing_age']
        assert (descriptor.listing_age is runtime_age) is (source == 'same_view')
        with descriptor.attach() as attached:
            actual_ages = attached.episode.listing_age
            expected_rows = np.searchsorted(episode.runtime.trade_dates, attached.episode.runtime.trade_dates)
            np.testing.assert_array_equal(actual_ages, ages[expected_rows])
            assert not actual_ages.flags.writeable
            assert_trace_equal(
                expected,
                run_day_config_episode(EpisodeSession(attached.episode), lambda _: config),
            )


def test_partial_creation_failure_releases_all_created_segments(lightweight_episode, monkeypatch):
    created = []

    def allocate(*args, **kwargs):
        if len(created) == 3:
            raise OSError('injected allocation failure')
        segment = SharedMemory(*args, **kwargs)
        created.append(segment)
        return segment

    monkeypatch.setattr(shared_episode, 'SharedMemory', allocate)
    with pytest.raises(OSError, match='injected allocation failure'):
        SharedPreparedEpisodeOwner.create(lightweight_episode)
    assert len(created) == 3
    for segment in created:
        assert segment._mmap is None
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=segment.name, create=False)


def test_partial_attach_failure_closes_every_handle_including_invalid_array(lightweight_episode, monkeypatch):
    opened = []

    def attach(*args, **kwargs):
        segment = SharedMemory(*args, **kwargs)
        opened.append(segment)
        return segment

    with SharedPreparedEpisodeOwner.create(lightweight_episode) as owner:
        descriptor = owner.descriptor
        # The invalid ndarray fails after opening its shared segment, following
        # several successful attachments that must also be closed.
        invalid = replace(descriptor.factor_raw, shape=(descriptor.factor_raw.nbytes + 1,))
        broken = replace(descriptor, factor_raw=invalid)
        monkeypatch.setattr(shared_episode, 'SharedMemory', attach)
        with pytest.raises(TypeError, match='buffer is too small'):
            broken.attach()
        assert len(opened) == 2 + len(descriptor.runtime_data)
        assert all(segment._mmap is None for segment in opened)
    for segment in opened:
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=segment.name, create=False)


def test_summary_runs_identical_daily_account_chain(tmp_path):
    path = tmp_path / 'runtime.npz'
    write_canonical_runtime(path)
    episode = prepare_episode_from_runtime(path, '2020-06-10', '2020-06-24',
                                          prefilter_n=25, encode_observations=False)
    schema = ActionSchema()
    rng = np.random.default_rng(3151)
    configs = [schema.decode(rng.uniform(-.99, 1, schema.action_dim))
               for _ in range(episode.transition_count)]
    def run(details):
        cursor = iter(configs)
        return run_day_config_episode(EpisodeSession(episode), lambda _: next(cursor), record_details=details)
    full, summary = run(True), run(False)
    for name in ('decision_dates', 'next_decision_dates', 'rewards', 'portfolio_returns',
                 'nav', 'cash', 'exposure', 'full_investment_contract'):
        np.testing.assert_array_equal(getattr(full, name), getattr(summary, name))
    assert summary.metrics == full.metrics
    assert summary.executed_sell_count == full.executed_sell_count
    assert summary.full_investment_contract_satisfied
    assert not hasattr(summary, 'fills')
