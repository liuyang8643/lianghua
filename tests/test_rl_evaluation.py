from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from ai.rl.evaluation import (
    BacktestRequest,
    FixedConfigProvider,
    FrozenPPOProvider,
    parallel_backtests_for_episode,
)
from ai.bundle import file_sha256
from env.action_schema import ActionSchema
from env.backtest import EpisodeSession, run_day_config_episode
from rl_test_data import build_episode, static_config, fit_normalizer


@pytest.fixture
def episode(tmp_path):
    return build_episode(tmp_path / "runtime.npz")


def test_backtest_request_has_no_window_or_slice_channel():
    names = {field.name for field in fields(BacktestRequest)}

    assert "episode_decision_slice" not in names
    assert "window" not in names


def test_static_parallel_benchmark_consumes_the_complete_episode(episode):
    schema = ActionSchema()
    config = static_config(schema)
    normalizer = fit_normalizer(episode)
    request = BacktestRequest(
        task_id="static",
        provider=FixedConfigProvider(schema.to_static_config(config)),
        action_schema_payload=schema.to_dict(),
        normalizer_payload=normalizer.to_dict(),
        initial_cash=1_000_000.0,
    )

    parallel = parallel_backtests_for_episode(
        episode,
        (request,),
        max_workers=1,
    )["static"]
    direct = run_day_config_episode(EpisodeSession(episode, action_schema=schema), lambda _: config)

    assert len(parallel.rewards) == episode.transition_count
    np.testing.assert_array_equal(parallel.nav, direct.nav)
    np.testing.assert_array_equal(parallel.rewards, direct.rewards)


def test_fixed_provider_requires_typed_config():
    with pytest.raises(TypeError, match="DayConfig"):
        FixedConfigProvider((2.0,))


def test_frozen_provider_validates_absolute_path_and_digest(tmp_path):
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"ppo")

    provider = FrozenPPOProvider(str(checkpoint.resolve()), file_sha256(checkpoint))
    assert provider.checkpoint_path == str(checkpoint.resolve())
    with pytest.raises(ValueError, match="SHA-256"):
        FrozenPPOProvider(str(checkpoint.resolve()), "bad")
