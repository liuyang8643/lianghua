"""Actual spawn and shared-memory cleanup using synthetic data, without learning."""

from multiprocessing import active_children, TimeoutError as MultiprocessingTimeoutError
from threading import get_ident

import pytest

from ai.rl.checkpoint import capture_evaluation_checkpoint
from ai.rl.evaluation import BacktestRequest, BacktestWorkerError, FrozenPPOProvider, FixedConfigProvider, parallel_backtests
from ai.rl.train import FrozenEvaluationQueue
from env.shared_episode import ResidentPreparedEpisode
from env.action_schema import ActionSchema
from rl_test_data import build_episode, static_config, fit_normalizer
from test_rl_frozen_evaluation import ModelBytes


@pytest.mark.parametrize("outcome", ["success", "worker_error", "timeout"])
def test_control_thread_joins_spawn_worker_before_shared_owner_release(tmp_path, outcome):
    episode = build_episode(tmp_path / "runtime.npz")
    schema = ActionSchema()
    normalizer = fit_normalizer(episode)
    snapshot = capture_evaluation_checkpoint(ModelBytes(), tmp_path)
    fixed = FixedConfigProvider(schema.to_static_config(static_config(schema)))
    provider = (FrozenPPOProvider(str(snapshot.path.resolve()), snapshot.sha256)
                if outcome == "worker_error" else fixed)
    request = BacktestRequest(task_id="lifecycle", provider=provider,
        action_schema_payload=schema.to_dict(), normalizer_payload=normalizer.to_dict(),
        initial_cash=1_000_000.0)
    previous_children = {child.pid for child in active_children()}
    consumed = []
    main_thread = get_ident()
    owner = ResidentPreparedEpisode(episode)

    def replay(frozen):
        assert get_ident() != main_thread
        frozen.validate()
        return parallel_backtests(owner.descriptor, (request,), max_workers=1,
            timeout_seconds=0.000001 if outcome == "timeout" else 30)[request.task_id]

    def consume(frozen, trace, elapsed):
        assert get_ident() == main_thread
        assert len(trace.rewards) == episode.transition_count
        consumed.append(trace)

    queue = FrozenEvaluationQueue(replay, consume)
    try:
        queue.submit(snapshot)
        if outcome == "success":
            assert queue.poll(wait=True)
            assert len(consumed) == 1
        else:
            error = BacktestWorkerError if outcome == "worker_error" else MultiprocessingTimeoutError
            with pytest.raises(error):
                queue.poll(wait=True)
            assert not consumed and snapshot.path.exists()
    finally:
        queue.close()
        assert {child.pid for child in active_children()} == previous_children
        owner.close()
    assert snapshot.path.exists() == (outcome != "success")
