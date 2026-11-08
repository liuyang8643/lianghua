"""No learner or market data: frozen persistence and asynchronous lifecycle contracts."""

import json
from pathlib import Path
from contextlib import ExitStack
from threading import Event, Timer, get_ident

import pytest

from ai.rl.checkpoint import (
    capture_evaluation_checkpoint, promote_evaluation_checkpoint,
    save_latest_train_checkpoint, atomic_write_json,
)
from ai.rl.train import FrozenEvaluationQueue, _validate_evaluation_completion, EVALUATION_COMPLETION_FILE


class ModelBytes:
    num_timesteps = 80
    _n_updates = 10
    wbr_run_identity_sha256 = "a" * 64
    wbr_run_identity_version = "test-identity"
    payload = b"frozen-policy-and-optimizer"

    def save(self, path):
        Path(path).write_bytes(self.payload)


def test_promotion_uses_frozen_bytes_and_counters_after_learner_advances(tmp_path):
    model = ModelBytes()
    snapshot = capture_evaluation_checkpoint(model, tmp_path)
    model.payload = b"later-policy"
    model.num_timesteps = 160
    model._n_updates = 20
    state = promote_evaluation_checkpoint(snapshot, tmp_path, file_name="selected.zip",
                                         role="selected", metrics={"calmar": 1.3})
    assert (tmp_path / "selected.zip").read_bytes() == b"frozen-policy-and-optimizer"
    assert (state["timesteps"], state["ppo_updates"]) == (80, 10)
    assert state["sha256"] == snapshot.sha256
    snapshot.release()
    assert not snapshot.path.exists()
    assert (tmp_path / "selected.zip").exists()


@pytest.mark.parametrize("tamper", ["bytes", "counter", "identity"])
def test_promotion_rejects_changed_snapshot(tmp_path, tamper):
    snapshot = capture_evaluation_checkpoint(ModelBytes(), tmp_path)
    if tamper == "bytes":
        snapshot.path.write_bytes(b"changed")
    else:
        sidecar = snapshot.path.with_suffix(".json")
        state = json.loads(sidecar.read_text())
        state["timesteps" if tamper == "counter" else "run_identity_sha256"] = 999
        atomic_write_json(sidecar, state)
    with pytest.raises(ValueError, match="changed"):
        promote_evaluation_checkpoint(snapshot, tmp_path, file_name="selected.zip",
                                      role="selected", metrics={})
    assert not (tmp_path / "selected.zip").exists()


def test_queue_retains_snapshot_until_main_thread_consumes_and_enforces_one_pending(tmp_path):
    main_thread = get_ident()
    entered, finish = Event(), Event()
    consumed = []
    snapshot = capture_evaluation_checkpoint(ModelBytes(), tmp_path)

    def replay(frozen):
        assert get_ident() != main_thread
        entered.set()
        assert finish.wait(5)
        frozen.validate()
        return "trace"

    def consume(frozen, trace, elapsed):
        assert get_ident() == main_thread
        assert frozen.path.exists()
        assert trace == "trace" and elapsed >= 0
        consumed.append(frozen.timesteps)

    queue = FrozenEvaluationQueue(replay, consume)
    try:
        queue.submit(snapshot)
        assert entered.wait(5)
        assert not queue.poll()
        with pytest.raises(RuntimeError, match="previous"):
            queue.submit(snapshot)
        finish.set()
        assert queue.poll(wait=True)
        assert consumed == [80]
        assert queue.pending is None and not snapshot.path.exists()
        assert not queue.poll(wait=True)
    finally:
        finish.set()
        queue.close()


@pytest.mark.parametrize("failure_at", ["replay", "consume"])
def test_queue_propagates_failure_and_preserves_snapshot(tmp_path, failure_at):
    snapshot = capture_evaluation_checkpoint(ModelBytes(), tmp_path)
    exited = Event()

    def replay(frozen):
        try:
            if failure_at == "replay":
                raise RuntimeError("worker failed")
            return "trace"
        finally:
            exited.set()

    def consume(*args):
        raise RuntimeError("selection failed")

    queue = FrozenEvaluationQueue(replay, consume)
    queue.submit(snapshot)
    try:
        with pytest.raises(RuntimeError, match="failed"):
            queue.poll(wait=True)
    finally:
        queue.close()
    assert exited.is_set() and snapshot.path.exists()
    snapshot.validate()


@pytest.mark.parametrize("state", ["missing", "pending", "other_identity", "old_latest", "complete"])
def test_overlap_resume_requires_drained_matching_canonical_checkpoint(tmp_path, state):
    model = ModelBytes()
    latest = save_latest_train_checkpoint(model, tmp_path,
        run_identity_sha256=model.wbr_run_identity_sha256,
        timesteps=model.num_timesteps, ppo_updates=model._n_updates, phase="drained")
    identity = {"identity_sha256": model.wbr_run_identity_sha256,
                "contract": {"algorithm": {"evaluation_execution": "overlap"}}}
    completion = {"complete": state != "pending",
                  "run_identity_sha256": identity["identity_sha256"], "latest": latest}
    if state == "other_identity":
        completion["run_identity_sha256"] = "b" * 64
    if state == "old_latest":
        completion["latest"] = {**latest, "timesteps": 0}
    if state != "missing":
        atomic_write_json(tmp_path / EVALUATION_COMPLETION_FILE, completion)
    if state == "complete":
        _validate_evaluation_completion(tmp_path, identity)
    else:
        with pytest.raises(ValueError, match="overlap resume"):
            _validate_evaluation_completion(tmp_path, identity)


def test_blocking_resume_does_not_require_overlap_completion(tmp_path):
    _validate_evaluation_completion(tmp_path, {"contract": {"algorithm": {}}})


def test_learner_failure_joins_pending_replay_before_owner_cleanup(tmp_path):
    snapshot = capture_evaluation_checkpoint(ModelBytes(), tmp_path)
    entered, finish = Event(), Event()
    order = []

    def replay(frozen):
        entered.set()
        assert finish.wait(5)
        order.append("worker_joined")
        return "trace"

    def consume(*args):
        pytest.fail("a failed learner must not silently select the pending candidate")

    timer = Timer(0.05, finish.set)
    try:
        with pytest.raises(RuntimeError, match="learner failed"):
            with ExitStack() as resources:
                resources.callback(lambda: order.append("owner_closed"))
                queue = FrozenEvaluationQueue(replay, consume)
                resources.callback(queue.close)
                queue.submit(snapshot)
                assert entered.wait(5)
                timer.start()
                raise RuntimeError("learner failed")
    finally:
        finish.set()
        timer.cancel()
        timer.join()
    assert order == ["worker_joined", "owner_closed"]
    assert snapshot.path.exists()
