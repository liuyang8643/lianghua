"""Precompute with full history, then project replay rows without recomputation."""
from dataclasses import replace
import json

import numpy as np
import pytest

from env.action_schema import ActionSchema
from env.backtest import EpisodeSession, prepare_episode_from_runtime, run_day_config_episode
from env.contracts import AccountState, PolicyHistory, PolicyMemory
from env.encoder import TrainOnlyNormalizer
from env.observation import ObservationBuilder
from env.shared_episode import SharedPreparedEpisodeOwner
from factor import precompute_factors
from factor.library.bilibili import calculate_bilibili_scores
from rl_test_data import write_runtime
from test_backtest_lightweight import assert_trace_equal


@pytest.fixture
def history_episode(tmp_path):
    path = tmp_path / "history.npz"
    write_runtime(path, end="2022-06-01")
    with np.load(path, allow_pickle=False) as data:
        values = {name: data[name] for name in data.files}
    # IPO and an old lifetime extreme lie outside the retained replay window.
    values["high"][10] *= 10
    np.savez_compressed(path, **values)
    return prepare_episode_from_runtime(path, "2022-05-01", "2022-05-10", lookback=504, prefilter_n=3)


def fit_normalizer(episode):
    return TrainOnlyNormalizer.fit(episode.market_store,episode.encoder.output_schema,
        dataset_role="train",initial_cash=1_000_000.0)


def test_compaction_preserves_every_field_factor_schema_and_normalizer(history_episode):
    source = history_episode
    original_manifest = source.runtime.manifest.as_dict()
    assert "replay_projection" not in original_manifest
    compact = source.compact_for_replay()
    assert compact.compact_for_replay() is compact
    assert compact.decision_start == compact.runtime.manifest.actual_preload_rows == 504
    assert compact.observation_count == source.observation_count
    assert compact.runtime.stock_codes == source.runtime.stock_codes
    assert compact.runtime.manifest.fields == source.runtime.manifest.fields
    assert set(compact.runtime.data) == set(source.runtime.data)
    proof = compact.runtime.manifest.replay_projection
    assert proof.source_manifest.as_dict() == original_manifest
    assert proof.source_row_offset == source.decision_start - 504 > 10
    assert compact.runtime.manifest.loaded_start == str(compact.runtime.trade_dates[0])
    assert compact.runtime.manifest.requested_preload_rows == source.runtime.manifest.requested_preload_rows
    for name, array in source.runtime.data.items():
        expected = array[proof.source_row_offset:source.decision_stop] if array.ndim == 2 else array
        np.testing.assert_array_equal(compact.runtime.field(name), expected)
        assert not compact.runtime.field(name).flags.writeable
    for name in ("raw", "ranks", "validity", "filters"):
        np.testing.assert_array_equal(getattr(compact.factors, name),
                                      getattr(source.factors, name)[proof.source_row_offset:source.decision_stop])
    assert compact.encoder.output_schema == source.encoder.output_schema
    assert compact.observation_builder.schema == source.observation_builder.schema
    assert fit_normalizer(compact).to_dict() == fit_normalizer(source).to_dict()
    # Recomputing after slicing would lose the IPO anchor and silently destroy
    # valid lifetime signals. The production precompute entry must reject it.
    factor = source.factors.factor_names.index("BiliHighLifetimeRangeRatio")
    assert np.isfinite(compact.factors.raw[-1, factor]).all()
    wrong = calculate_bilibili_scores(compact.runtime.trade_dates, compact.runtime.data)
    assert np.isnan(wrong["BiliHighLifetimeRangeRatio"][-1]).all()
    with pytest.raises(ValueError, match="reload the original runtime"):
        precompute_factors(compact.runtime)
    assert source.runtime.manifest.as_dict() == original_manifest


@pytest.mark.parametrize("populated_history", [False, True])
def test_shared_compaction_keeps_raw_cached_account_and_complete_trace_exact(history_episode, populated_history):
    source = history_episode
    schema = ActionSchema()
    config = schema.decode(np.zeros(schema.action_dim))
    normalizer = fit_normalizer(source)
    original_trace = run_day_config_episode(EpisodeSession(source, normalizer=normalizer), lambda _: config)
    with SharedPreparedEpisodeOwner.create(source) as owner:
        with owner.descriptor.attach() as attached:
            compact = attached.episode
            actual_trace = run_day_config_episode(EpisodeSession(compact, normalizer=normalizer), lambda _: config)
            assert_trace_equal(original_trace, actual_trace)
            left, right = [EpisodeSession(item, normalizer=normalizer, include_critic_context=True)
                           for item in (source, compact)]
            reset = {}
            if populated_history:
                dates = source.runtime.trade_dates[source.decision_start - 504:source.decision_start]
                values = np.tile(np.r_[schema.encode(config).astype(np.float64), .1, .001], (504, 1))
                memory = PolicyMemory(config, .1, .001, PolicyHistory(dates, values, schema.schema_hash))
                reset = {"account": AccountState(cash=1_000_000., nav=1_000_000., peak_nav=1_000_000.),
                         "policy_memory": memory}
            left.reset(**reset)
            right.reset(**reset)
            while True:
                lraw, rraw = left.current_observation, right.current_observation
                assert lraw.schema_version == rraw.schema_version
                assert lraw.decision_date == rraw.decision_date
                for field in ("stock_panel", "position_panel", "portfolio",
                              "policy_history", "time_mask", "pit_universe_mask"):
                    np.testing.assert_array_equal(getattr(lraw, field), getattr(rraw, field))
                np.testing.assert_array_equal(left.encoded_observation, right.encoded_observation)
                if left.terminated:
                    assert right.terminated
                    break
                action = np.r_[np.sin(left.encoded_observation[:len(schema.factor_names)]), [0.]]
                current = schema.decode(action)
                lt, rt = left.step(current), right.step(current)
                assert lt.order_plan == rt.order_plan
                assert lt.step_result.fills == rt.step_result.fills
                assert lt.step_result.account_state == rt.step_result.account_state
                assert lt.step_result.reward == rt.step_result.reward
                assert lt.step_result.policy_memory == rt.step_result.policy_memory
            assert owner.descriptor.runtime_manifest.replay_projection == compact.runtime.manifest.replay_projection


def test_compaction_rejects_changed_provenance(history_episode):
    compact = history_episode.compact_for_replay()
    proof = compact.runtime.manifest.replay_projection
    wrong = replace(proof, factor_schema_hash="f" * 64)
    runtime = replace(compact.runtime, manifest=replace(compact.runtime.manifest, replay_projection=wrong))
    with pytest.raises(ValueError, match="provenance differs"):
        replace(compact, runtime=runtime)
    wrong = replace(proof, observation_schema="wrong")
    runtime = replace(compact.runtime, manifest=replace(compact.runtime.manifest, replay_projection=wrong))
    with pytest.raises(ValueError, match="observation or encoder schema"):
        replace(compact, runtime=runtime)


def test_shared_bytes_materialize_only_retained_rows(history_episode):
    source = history_episode
    with SharedPreparedEpisodeOwner.create(source) as owner:
        descriptor = owner.descriptor
        source_rows = source.runtime.n_dates
        retained_rows = descriptor.runtime_trade_dates.shape[0]
        assert retained_rows == 504 + source.observation_count
        assert retained_rows < source_rows
        unprojected_bytes = 0
        for array in descriptor.arrays:
            unprojected_bytes += (array.nbytes * source_rows // retained_rows
                                  if array.shape[0] == retained_rows and array.label != "episode.raw_rows"
                                  else array.nbytes)
        # Raw observations now retain the complete stock/date fields once;
        # their fixed store adds bytes but does not duplicate rolling windows.
        assert descriptor.shared_memory_bytes < unprojected_bytes
        assert descriptor.raw_rows.shape[0] == 503 + source.observation_count
        print(json.dumps({"before_shared_bytes": unprojected_bytes,
                          "after_shared_bytes": descriptor.shared_memory_bytes,
                          "original_rows": source_rows, "retained_rows": retained_rows}))
