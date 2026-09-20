"""Spawn-safe orchestration for independent frozen-policy backtests.

Each worker attaches one already-sealed :class:`PreparedEpisode` through
read-only shared memory and then executes :func:`env.backtest.run_episode`
serially.  This module parallelizes independent account chains only; it never
splits or reorders the transitions inside one account chain.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import math
from multiprocessing import TimeoutError as MultiprocessingTimeoutError
from multiprocessing import current_process
from multiprocessing import get_context
from multiprocessing.connection import Connection, wait
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
import os
from pathlib import Path
import time
import traceback
from typing import Mapping, TypeAlias

from ai.bundle import file_sha256
from ai.rl.policy import predict_encoded_action
from ai.rl.device import load_cuda_ppo, require_cuda_model
from ai.rl.rollout import (
    configure_single_thread_runtime,
)
import numpy as np
from numpy.typing import NDArray
from stable_baselines3 import PPO

from env.action_schema import ActionSchema
from env.backtest import PreparedEpisode, RolloutTrace, EpisodeSession, run_day_config_episode, run_episode
from env.encoder import RawMarketStore, TrainOnlyNormalizer
from env.gym_adapter import WBRGymEnv
from env.shared_episode import (
    AttachedPreparedEpisode,
    SharedPreparedEpisodeDescriptor,
    SharedPreparedEpisodeOwner,
)


EVALUATION_START_METHOD = "spawn"
EVALUATION_PROTOCOL_VERSION = (
    "wbr-frozen-offline-backtest-evaluation-v6-periodic-three-split"
)
DEFAULT_EVALUATION_TIMEOUT_SECONDS = 3_600.0
MAX_PARALLEL_BACKTEST_WORKERS = 64
_PROCESS_STOP_GRACE_SECONDS = 5.0
_SHA256_HEX_LENGTH = 64


@dataclass(frozen=True, slots=True)
class FrozenPPOProvider:
    """One immutable PPO artifact, verified before and after worker loading."""

    checkpoint_path: str
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        path = Path(self.checkpoint_path)
        if not path.is_absolute():
            raise ValueError("frozen PPO checkpoint path must be absolute")
        digest = str(self.checkpoint_sha256).lower()
        if len(digest) != _SHA256_HEX_LENGTH or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("frozen PPO checkpoint SHA-256 is invalid")
        object.__setattr__(self, "checkpoint_path", str(path))
        object.__setattr__(self, "checkpoint_sha256", digest)


@dataclass(frozen=True, slots=True)
class FixedConfigProvider:
    """An exact static DayConfig; no float32 action roundtrip."""

    config: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.config, dict):
            raise TypeError("fixed provider requires a serialized DayConfig dictionary")


BacktestProvider: TypeAlias = FrozenPPOProvider | FixedConfigProvider


@dataclass(frozen=True, slots=True)
class BacktestRequest:
    """Small pickleable description of one independent account rollout.

    Runtime and holdout paths are deliberately absent.  The caller supplies a
    sealed shared-memory descriptor separately to :func:`parallel_backtests`.
    """

    task_id: str
    provider: BacktestProvider
    action_schema_payload: Mapping[str, object]
    normalizer_payload: Mapping[str, object] | None
    initial_cash: float

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("backtest task_id must be a non-empty string")
        if not isinstance(
            self.provider,
            (FrozenPPOProvider, FixedConfigProvider),
        ):
            raise TypeError("unsupported backtest provider")
        if not math.isfinite(self.initial_cash) or self.initial_cash <= 0.0:
            raise ValueError("backtest initial_cash must be finite and positive")


@dataclass(frozen=True, slots=True)
class _WorkerBacktestTask:
    descriptor: SharedPreparedEpisodeDescriptor
    request: BacktestRequest


@dataclass(frozen=True, slots=True)
class _WorkerMessage:
    task_id: str
    trace: RolloutTrace | None
    error_type: str | None
    error_message: str | None
    error_traceback: str | None


class BacktestWorkerError(RuntimeError):
    """A spawn worker failed after the parent had dispatched its task."""


class _FrozenPPOCallable:
    def __init__(self, model: PPO, action_schema: ActionSchema) -> None:
        require_cuda_model(model)
        expected_shape = (action_schema.action_dim,)
        if model.action_space.shape != expected_shape:
            raise ValueError(
                "frozen PPO action dimension differs from ActionSchema: "
                f"{model.action_space.shape} != {expected_shape}"
            )
        typed_schema = model.policy.typed_action_schema
        if typed_schema.schema_hash != action_schema.schema_hash:
            raise ValueError("frozen PPO typed ActionSchema differs from evaluation schema")
        self.model = model

    def __call__(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        return predict_encoded_action(self.model, observation, deterministic=True)


def _load_frozen_ppo(
    provider: FrozenPPOProvider,
    action_schema: ActionSchema,
    market_store: RawMarketStore,
    normalizer: TrainOnlyNormalizer,
) -> _FrozenPPOCallable:
    path = Path(provider.checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"frozen PPO checkpoint is missing: {path}")
    before = file_sha256(path)
    if before != provider.checkpoint_sha256:
        raise ValueError("frozen PPO checkpoint SHA-256 mismatch before load")
    model = load_cuda_ppo(path)
    model.policy.bind_market_store(market_store, normalizer)
    after = file_sha256(path)
    if after != provider.checkpoint_sha256:
        raise RuntimeError("frozen PPO checkpoint changed while it was loaded")
    return _FrozenPPOCallable(model, action_schema)


def _close_worker_episode(
    env: WBRGymEnv | None,
    attached: AttachedPreparedEpisode,
) -> list[BaseException]:
    failures: list[BaseException] = []
    if env is not None:
        try:
            env.close()
        except BaseException as error:
            failures.append(error)
        # Drop every ndarray owner before closing SharedMemory mappings.
        env.episode = None  # type: ignore[assignment]
    gc.collect()
    try:
        attached.close()
    except BaseException as error:
        failures.append(error)
    return failures


def _run_worker_backtest(task: _WorkerBacktestTask) -> tuple[str, RolloutTrace]:
    """Top-level spawn target; one invocation owns one serial account chain."""

    configure_single_thread_runtime()
    attached = task.descriptor.attach()
    env: WBRGymEnv | None = None
    try:
        request = task.request
        action_schema = ActionSchema.from_dict(request.action_schema_payload)
        normalizer = (
            None
            if request.normalizer_payload is None
            else TrainOnlyNormalizer.from_dict(request.normalizer_payload)
        )
        if isinstance(request.provider, FixedConfigProvider):
            config = action_schema.from_serialized_day_config(request.provider.config)
            session = EpisodeSession(attached.episode, initial_cash=request.initial_cash,
                                     action_schema=action_schema)
            trace = run_day_config_episode(session, lambda _: config)
        else:
            env = WBRGymEnv(attached.episode, action_schema=action_schema,
                            normalizer=normalizer, initial_cash=request.initial_cash)
            if normalizer is None:
                raise ValueError("raw PPO replay requires its frozen training normalizer")
            trace = run_episode(env, _load_frozen_ppo(
                request.provider, action_schema, attached.episode.market_store, normalizer,
            ))
    except BaseException as execution_error:
        cleanup_failures = _close_worker_episode(env, attached)
        if cleanup_failures:
            raise BaseExceptionGroup(
                "backtest execution and worker cleanup both failed",
                [execution_error, *cleanup_failures],
            )
        raise
    cleanup_failures = _close_worker_episode(env, attached)
    if cleanup_failures:
        raise BaseExceptionGroup(
            "backtest worker cleanup failed",
            cleanup_failures,
        )
    return task.request.task_id, trace


def _worker_process_entry(
    task: _WorkerBacktestTask,
    result_connection: Connection,
) -> None:
    """Non-daemon spawn entry that always sends a serializable outcome."""

    if current_process().daemon:
        raise RuntimeError("backtest workers must not be daemon processes")
    try:
        task_id, trace = _run_worker_backtest(task)
        message = _WorkerMessage(
            task_id=task_id,
            trace=trace,
            error_type=None,
            error_message=None,
            error_traceback=None,
        )
    except BaseException as error:
        message = _WorkerMessage(
            task_id=task.request.task_id,
            trace=None,
            error_type=type(error).__name__,
            error_message=str(error),
            error_traceback="".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
        )
    try:
        result_connection.send(message)
    finally:
        result_connection.close()


def _validate_batch(
    descriptor: SharedPreparedEpisodeDescriptor,
    requests: tuple[BacktestRequest, ...],
) -> None:
    if not requests:
        raise ValueError("backtest requests must not be empty")
    task_ids = tuple(request.task_id for request in requests)
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("backtest task_ids must be unique")
    if descriptor.runtime_manifest.source_path != "<shared-memory>":
        raise ValueError("backtest descriptor must not expose a runtime path")
    for request in requests:
        schema = ActionSchema.from_dict(request.action_schema_payload)
        if request.normalizer_payload is not None:
            normalizer = TrainOnlyNormalizer.from_dict(request.normalizer_payload)
            if normalizer.encoder_schema != descriptor.raw_encoder_schema:
                raise ValueError("backtest normalizer and descriptor schemas differ")
        provider = request.provider
        if isinstance(provider, FixedConfigProvider):
            schema.from_serialized_day_config(provider.config)


def _ordered_results(
    requests: tuple[BacktestRequest, ...],
    results: list[tuple[str, RolloutTrace]],
) -> dict[str, RolloutTrace]:
    expected = tuple(request.task_id for request in requests)
    actual = tuple(task_id for task_id, _ in results)
    if actual != expected:
        raise RuntimeError(
            f"backtest result order or identity changed: {actual} != {expected}"
        )
    return {
        task_id: trace
        for task_id, trace in results
    }


def _stop_processes(
    processes: list[BaseProcess],
    connections: list[Connection],
) -> list[BaseException]:
    failures: list[BaseException] = []
    for connection in connections:
        try:
            connection.close()
        except BaseException as error:
            failures.append(error)
    for process in processes:
        try:
            if process.is_alive():
                process.terminate()
        except BaseException as error:
            failures.append(error)
    deadline = time.monotonic() + _PROCESS_STOP_GRACE_SECONDS
    for process in processes:
        try:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        except BaseException as error:
            failures.append(error)
    for process in processes:
        try:
            if process.is_alive():
                process.kill()
        except BaseException as error:
            failures.append(error)
    deadline = time.monotonic() + _PROCESS_STOP_GRACE_SECONDS
    for process in processes:
        try:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
            if process.is_alive():
                failures.append(
                    RuntimeError(
                        f"backtest worker pid={process.pid} did not stop"
                    )
                )
            process.close()
        except BaseException as error:
            failures.append(error)
    return failures


def _join_completed_processes(
    processes: list[BaseProcess],
    connections: list[Connection],
) -> None:
    for connection in connections:
        connection.close()
    deadline = time.monotonic() + _PROCESS_STOP_GRACE_SECONDS
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            raise RuntimeError(
                f"completed backtest worker pid={process.pid} did not exit"
            )
        if process.exitcode != 0:
            raise RuntimeError(
                "backtest worker exited abnormally after returning a result: "
                f"pid={process.pid}, exitcode={process.exitcode}"
            )
    for process in processes:
        process.close()


def _run_spawn_wave(
    context: BaseContext,
    tasks: tuple[_WorkerBacktestTask, ...],
    *,
    deadline: float,
) -> list[tuple[str, RolloutTrace]]:
    processes: list[BaseProcess] = []
    receive_connections: list[Connection] = []
    send_connections: list[Connection] = []
    try:
        for index, task in enumerate(tasks):
            receive_connection, send_connection = context.Pipe(duplex=False)
            receive_connections.append(receive_connection)
            send_connections.append(send_connection)
            process = context.Process(
                target=_worker_process_entry,
                args=(task, send_connection),
                name=f"wbr-backtest-{index}-{task.request.task_id}",
            )
            process.daemon = False
            process.start()
            processes.append(process)
            send_connection.close()

        messages: dict[Connection, _WorkerMessage] = {}
        pending = set(receive_connections)
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise MultiprocessingTimeoutError(
                    "parallel backtest exceeded its finite timeout"
                )
            ready = wait(tuple(pending), timeout=remaining)
            if not ready:
                raise MultiprocessingTimeoutError(
                    "parallel backtest exceeded its finite timeout"
                )
            for connection in ready:
                try:
                    message = connection.recv()
                except EOFError as error:
                    raise RuntimeError(
                        "backtest worker exited without returning a result"
                    ) from error
                if not isinstance(message, _WorkerMessage):
                    raise RuntimeError("backtest worker returned an invalid payload")
                messages[connection] = message
                pending.remove(connection)

        ordered_messages = [messages[item] for item in receive_connections]
        for message in ordered_messages:
            if message.error_type is not None:
                raise BacktestWorkerError(
                    f"task {message.task_id!r} failed with {message.error_type}: "
                    f"{message.error_message}\n{message.error_traceback}"
                )
            if message.trace is None:
                raise RuntimeError("successful backtest worker omitted its trace")
        _join_completed_processes(processes, receive_connections)
        receive_connections = []
        return [
            (message.task_id, message.trace)
            for message in ordered_messages
            if message.trace is not None
        ]
    except BaseException as execution_error:
        cleanup_failures = _stop_processes(processes, receive_connections)
        for connection in send_connections:
            try:
                connection.close()
            except BaseException as error:
                cleanup_failures.append(error)
        if cleanup_failures:
            raise BaseExceptionGroup(
                "parallel backtest and process cleanup both failed",
                [execution_error, *cleanup_failures],
            )
        raise


def parallel_backtests(
    descriptor: SharedPreparedEpisodeDescriptor,
    requests: tuple[BacktestRequest, ...],
    *,
    max_workers: int | None = None,
    timeout_seconds: float = DEFAULT_EVALUATION_TIMEOUT_SECONDS,
) -> dict[str, RolloutTrace]:
    """Run independent complete backtests in a fail-closed spawn pool.

    The descriptor owner remains the caller's responsibility and must outlive
    this call.  :func:`parallel_backtests_for_episode` is the convenience API
    when no existing shared owner is available.
    """

    _validate_batch(descriptor, requests)
    if max_workers is None:
        worker_count = min(
            len(requests),
            os.cpu_count() or 1,
            MAX_PARALLEL_BACKTEST_WORKERS,
        )
    else:
        if (
            type(max_workers) is not int
            or not 1 <= max_workers <= MAX_PARALLEL_BACKTEST_WORKERS
        ):
            raise ValueError(
                "max_workers must be an int within "
                f"[1, {MAX_PARALLEL_BACKTEST_WORKERS}]"
            )
        worker_count = min(max_workers, len(requests))
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be finite and positive")

    context = get_context(EVALUATION_START_METHOD)
    tasks = tuple(_WorkerBacktestTask(descriptor, request) for request in requests)
    deadline = time.monotonic() + float(timeout_seconds)
    results: list[tuple[str, RolloutTrace]] = []
    for start in range(0, len(tasks), worker_count):
        results.extend(
            _run_spawn_wave(
                context,
                tasks[start : start + worker_count],
                deadline=deadline,
            )
        )
    return _ordered_results(requests, results)


def parallel_backtests_for_episode(
    episode: PreparedEpisode,
    requests: tuple[BacktestRequest, ...],
    *,
    max_workers: int | None = None,
    timeout_seconds: float = DEFAULT_EVALUATION_TIMEOUT_SECONDS,
) -> dict[str, RolloutTrace]:
    """Create, use, and always unlink a temporary shared episode owner."""

    owner = SharedPreparedEpisodeOwner.create(episode)
    try:
        results = parallel_backtests(
            owner.descriptor,
            requests,
            max_workers=max_workers,
            timeout_seconds=timeout_seconds,
        )
    except BaseException as execution_error:
        try:
            owner.close()
        except BaseException as cleanup_error:
            raise BaseExceptionGroup(
                "parallel backtest and shared owner cleanup both failed",
                [execution_error, cleanup_error],
            )
        raise
    owner.close()
    return results


__all__ = [
    "BacktestProvider",
    "BacktestRequest",
    "BacktestWorkerError",
    "DEFAULT_EVALUATION_TIMEOUT_SECONDS",
    "EVALUATION_PROTOCOL_VERSION",
    "EVALUATION_START_METHOD",
    "FixedConfigProvider",
    "FrozenPPOProvider",
    "MAX_PARALLEL_BACKTEST_WORKERS",
    "MultiprocessingTimeoutError",
    "parallel_backtests",
    "parallel_backtests_for_episode",
]
