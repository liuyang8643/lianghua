"""Explicit research factors reuse the production account timeline."""

from dataclasses import replace
import hashlib
import inspect

import numpy as np
import pytest

from env.action_schema import ActionSchema
from env.backtest import (
    EpisodeSession, PreparedEpisode, required_runtime_preload_rows,
    run_day_config_episode, run_episode, run_policy_episode,
)
from env.gym_adapter import WBRGymEnv
from factor import FactorDefinition, FactorMetadata, PRODUCTION_FACTORS, precompute_factors
from offline_data import load_runtime_slice
from test_backtest_lightweight import assert_trace_equal, current4_payload, write_canonical_runtime


class PriorClose:
    """Fixture factor: negative completed close, with no production alias."""

    def calc_batch(self, panel):
        result = np.full(panel["close"].shape, np.nan)
        result[1:] = -panel["close"][:-1]
        return result


DEFINITION = FactorDefinition(
    FactorMetadata(
        name="ResearchPriorClose", version="test-v1", hist_days=1,
        required_fields=("close",),
        implementation_hash=hashlib.sha256(inspect.getsource(PriorClose).encode()).hexdigest(),
    ),
    PriorClose,
)


@pytest.fixture
def runtime(tmp_path):
    path = tmp_path / "runtime.npz"
    write_canonical_runtime(path)
    return load_runtime_slice(
        path, "2020-06-20", "2020-06-24",
        preload_rows=required_runtime_preload_rows(64),
    )


def test_production_observation_modes_have_identical_account_traces(runtime):
    factors = precompute_factors(runtime)
    explicit = precompute_factors(runtime, definitions=PRODUCTION_FACTORS)
    assert factors.schema_hash == explicit.schema_hash
    assert factors.schema_version == explicit.schema_version
    for field in ("raw", "ranks", "validity", "filters"):
        np.testing.assert_array_equal(getattr(factors, field), getattr(explicit, field))
    observed = PreparedEpisode.build(runtime, factors, lookback=64, prefilter_n=25)
    research = PreparedEpisode.build(
        runtime, factors, prefilter_n=25, encode_observations=False,
    )
    config = ActionSchema().from_static_config(current4_payload()["individual_config"])
    left = EpisodeSession(observed)
    right = EpisodeSession(research)
    left_trace = run_day_config_episode(left, lambda _: config)
    right_trace = run_day_config_episode(right, lambda _: config)
    assert_trace_equal(left_trace, right_trace)
    assert left.current_account == right.current_account
    assert replace(left.current_policy_memory, history=None) == right.current_policy_memory
    assert right.current_policy_memory.history is None
    assert len(left.current_policy_memory.history.decision_dates) == observed.transition_count
    assert left.reward_state == right.reward_state
    assert right.last_transition.observation.shape == (0,)
    assert observed.market_store is not None
    assert research.market_store is None


def test_custom_vocabulary_uses_research_identity_and_canonical_execution(runtime):
    factors = precompute_factors(runtime, definitions=(DEFINITION,))
    assert factors.factor_names == ("ResearchPriorClose",)
    assert factors.schema_version.startswith("wbr.research-factors.")
    index = runtime.decision_start
    np.testing.assert_array_equal(factors.raw[index, 0], -runtime.field("close")[index - 1])
    assert not factors.raw.flags.writeable
    episode = PreparedEpisode.build(runtime, factors, encode_observations=False, prefilter_n=25)
    session = EpisodeSession(episode)
    assert session.action_schema.schema_version.startswith("day-config-research-")
    config = session.action_schema.decode(np.zeros(session.action_schema.action_dim))
    trace = run_day_config_episode(session, lambda _: config)
    assert len(trace.rewards) == episode.transition_count
    assert trace.full_investment_contract_satisfied
    assert all(np.isfinite(trace.nav))
    assert any(trace.fills)
    assert set(trace.day_configs[0]["weights"]) == {"ResearchPriorClose"}
    with pytest.raises(ValueError, match="research factor schema"):
        PreparedEpisode.build(runtime, factors)
    with pytest.raises(ValueError, match="distinct action schema"):
        EpisodeSession(episode, action_schema=ActionSchema(factor_names=factors.factor_names))


def test_no_observation_mode_rejects_model_consumers(runtime):
    episode = PreparedEpisode.build(
        runtime, precompute_factors(runtime), encode_observations=False,
    )
    for kwargs in ({"normalizer": object()}, {"include_critic_context": True}):
        with pytest.raises(ValueError, match="without observations"):
            EpisodeSession(episode, **kwargs)
    session = EpisodeSession(episode)
    observation, _ = session.reset()
    assert observation.shape == (0,) and observation.dtype == np.float32
    for name in ("current_observation", "observation_dimension", "critic_context_dimension"):
        with pytest.raises(ValueError, match="without observations"):
            getattr(session, name)
    with pytest.raises(ValueError, match="without observations"):
        WBRGymEnv(episode)
    with pytest.raises(ValueError, match="without observations"):
        run_episode(session, lambda _: np.zeros(10))
    with pytest.raises(ValueError, match="without observations"):
        run_policy_episode(session, object())


def test_research_definitions_and_preload_are_explicit(runtime):
    with pytest.raises(TypeError, match="non-empty tuple"):
        precompute_factors(runtime, definitions=())
    with pytest.raises(ValueError, match="unique"):
        precompute_factors(runtime, definitions=(DEFINITION, DEFINITION))
    missing = replace(DEFINITION, metadata=replace(DEFINITION.metadata, required_fields=("absent",)))
    with pytest.raises(ValueError, match="missing factor fields"):
        precompute_factors(runtime, definitions=(missing,))
    insufficient = replace(runtime, manifest=replace(runtime.manifest, requested_preload_rows=0))
    factors = precompute_factors(insufficient, definitions=(DEFINITION,))
    with pytest.raises(ValueError, match="preload"):
        PreparedEpisode.build(insufficient, factors, encode_observations=False)
