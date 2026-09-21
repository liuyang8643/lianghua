"""Deterministic vector-environment assembly for PPO rollouts.

PPO keeps one learner process.  When subprocess rollouts are requested, the
large immutable episode arrays live in ``multiprocessing.shared_memory`` and
each spawn worker reconstructs the public domain contracts from a small,
pickleable descriptor.  No worker reloads the runtime or recomputes factors.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing.connection import Connection, wait
import time
from typing import Any, Literal, Mapping
import warnings

# Sets the single-thread environment variables before NumPy/Torch import.
from ai.rl.rollout_worker import (
    CloudpickleWrapper,
    RolloutEnvFactory,
    RolloutWorkerAssignment,
    SINGLE_THREAD_ENVIRONMENT,
    WorkerFailurePayload as _WorkerFailurePayload,
    WorkerWBRGymEnv as _WorkerWBRGymEnv,
    build_worker_assignments,
    configure_worker_single_thread,
    exception_reporting_worker as _exception_reporting_worker,
)

import numpy as np
from numpy.typing import NDArray
import gymnasium as gym
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices
from stable_baselines3.common.vec_env.subproc_vec_env import _stack_obs

from env.action_schema import ActionSchema
from env.encoder import TrainOnlyNormalizer
from env.backtest import PreparedEpisode
from env.fees import DEFAULT_FEE_SCHEDULE, FeeSchedule
from env.gym_adapter import WBRGymEnv
from env.shared_episode import (
    SharedPreparedEpisodeDescriptor,
    SharedPreparedEpisodeOwner,
)


RolloutBackend = Literal["auto", "dummy", "subproc"]
ResolvedRolloutBackend = Literal["dummy", "subproc"]
ROLLOUT_SCHEMA_VERSION = "wbr-ppo-rollout-v5-configurable-training-episode"
ROLLOUT_START_METHOD = "spawn"
ROLLOUT_RESPONSE_TIMEOUT_SECONDS = 120.0
ROLLOUT_CLOSE_PHASE_TIMEOUT_SECONDS = 2.0


def configure_single_thread_runtime() -> None:
    """Pin native math and Torch execution to one thread in the learner process."""

    configure_worker_single_thread()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Torch only permits changing this value before inter-op work.  A
            # spawned rollout worker is fresh; the parent train entry already
            # pins it before creating the vector environment.
            if torch.get_num_interop_threads() != 1:
                raise


def resolve_rollout_backend(
    requested: str,
    n_envs: int,
) -> ResolvedRolloutBackend:
    """Resolve the CLI backend without silently changing explicit choices."""

    if requested not in {"auto", "dummy", "subproc"}:
        raise ValueError("rollout backend must be one of: auto, dummy, subproc")
    if type(n_envs) is not int or n_envs <= 0:
        raise ValueError("n_envs must be a positive int")
    if requested == "auto":
        return "subproc" if n_envs > 1 else "dummy"
    return requested  # type: ignore[return-value]


class RolloutWorkerError(RuntimeError):
    """Base error for a failed subprocess rollout command."""


class RolloutWorkerRemoteError(RolloutWorkerError):
    """An environment command raised inside one rollout worker."""

    def __init__(
        self,
        *,
        worker_index: int,
        command: str,
        failure: _WorkerFailurePayload,
    ) -> None:
        self.worker_index = worker_index
        self.command = command
        self.remote_exception_type = failure.exception_type
        self.remote_message = failure.message
        self.remote_traceback = failure.traceback
        super().__init__(
            f"rollout worker {worker_index} raised {failure.exception_type} "
            f"during {command}: {failure.message}\n"
            f"Remote traceback:\n{failure.traceback}"
        )


class RolloutWorkerTimeoutError(RolloutWorkerError):
    """One or more rollout workers exceeded the finite response deadline."""

    def __init__(
        self,
        *,
        command: str,
        pending_worker_indices: tuple[int, ...],
        timeout_seconds: float,
    ) -> None:
        self.command = command
        self.pending_worker_indices = pending_worker_indices
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"rollout workers {pending_worker_indices} timed out during {command} "
            f"after {timeout_seconds:g}s"
        )


class RolloutWorkerCommunicationError(RolloutWorkerError):
    """A worker pipe failed before a structured response arrived."""

    def __init__(
        self,
        *,
        worker_index: int | None,
        command: str,
        detail: str,
    ) -> None:
        self.worker_index = worker_index
        self.command = command
        self.detail = detail
        worker = "unknown" if worker_index is None else str(worker_index)
        super().__init__(
            f"rollout worker {worker} communication failed during {command}: {detail}"
        )


class RolloutWorkerProtocolError(RolloutWorkerError):
    """A worker returned a value that violates the SB3 VecEnv protocol."""


class _DeterministicSubprocVecEnv(SubprocVecEnv):
    """SB3 SubprocVecEnv with bounded replies and fail-closed ownership."""

    def __init__(
        self,
        env_fns: Sequence[Callable[[], gym.Env]],
        start_method: str | None = None,
        *,
        response_timeout_seconds: float = ROLLOUT_RESPONSE_TIMEOUT_SECONDS,
        resource_owner: SharedPreparedEpisodeOwner | None = None,
    ) -> None:
        self.waiting = False
        self.closed = False
        self.remotes: tuple[Connection, ...] = ()
        self.work_remotes: tuple[Connection, ...] = ()
        self.processes: list[mp.Process] = []
        self._resource_owner = resource_owner
        self._response_timeout_seconds = float(response_timeout_seconds)
        try:
            if not env_fns:
                raise ValueError("subprocess rollout requires at least one environment")
            if (
                not np.isfinite(self._response_timeout_seconds)
                or self._response_timeout_seconds <= 0.0
            ):
                raise ValueError("rollout response timeout must be finite and positive")
            context = mp.get_context(start_method or ROLLOUT_START_METHOD)
            pipes = tuple(context.Pipe() for _ in env_fns)
            self.remotes, self.work_remotes = zip(*pipes, strict=True)
            for work_remote, remote, env_fn in zip(
                self.work_remotes,
                self.remotes,
                env_fns,
                strict=True,
            ):
                process = context.Process(
                    target=_exception_reporting_worker,
                    args=(work_remote, remote, CloudpickleWrapper(env_fn)),
                    daemon=True,
                )
                process.start()
                self.processes.append(process)
                work_remote.close()

            spaces_result = self._request(
                (self.remotes[0],),
                (("get_spaces", None),),
                command="get_spaces",
            )[0]
            try:
                observation_space, action_space = spaces_result
            except (TypeError, ValueError) as error:
                self._abort_and_raise(
                    RolloutWorkerProtocolError(
                        "rollout worker 0 returned an invalid get_spaces response"
                    ),
                    cause=error,
                )
            VecEnv.__init__(self, len(env_fns), observation_space, action_space)
        except BaseException as startup_error:
            if not self.closed:
                failures = self._close_process_resources(send_close=False)
                failures.extend(self._close_resource_owner())
                self.waiting = False
                self.closed = True
                self._annotate_cleanup_failures(startup_error, failures)
            raise

    @property
    def response_timeout_seconds(self) -> float:
        return self._response_timeout_seconds

    def step_async(self, actions: np.ndarray) -> None:
        if self.closed:
            raise RuntimeError("subprocess rollout environment is closed")
        requests = tuple(
            ("step", action)
            for action in actions
        )
        if len(requests) != len(self.remotes):
            raise ValueError("action batch does not match rollout worker count")
        self._send(self.remotes, requests, command="step")
        self.waiting = True

    def step_wait(
        self,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_], list[dict]]:
        if not self.waiting:
            raise RuntimeError("step_wait called without a pending rollout step")
        results = self._receive(self.remotes, command="step")
        self.waiting = False
        try:
            observations, rewards, dones, infos, reset_infos = zip(
                *results,
                strict=True,
            )
            self.reset_infos = list(reset_infos)
            return (
                _stack_obs(observations, self.observation_space),
                np.asarray(rewards, dtype=np.float32),
                np.stack(dones),
                infos,
            )
        except BaseException as error:
            self._abort_and_raise(
                RolloutWorkerProtocolError(
                    "subprocess rollout returned an invalid step response"
                ),
                cause=error,
            )

    def reset(self) -> NDArray[np.float32]:
        requests = tuple(
            ("reset", (self._seeds[index], self._options[index]))
            for index in range(len(self.remotes))
        )
        results = self._request(self.remotes, requests, command="reset")
        try:
            observations, reset_infos = zip(*results, strict=True)
            self.reset_infos = list(reset_infos)
            result = _stack_obs(observations, self.observation_space)
        except BaseException as error:
            self._abort_and_raise(
                RolloutWorkerProtocolError(
                    "subprocess rollout returned an invalid reset response"
                ),
                cause=error,
            )
        self._reset_seeds()
        self._reset_options()
        return result

    def get_images(self) -> Sequence[np.ndarray | None]:
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes "
                "rgb_array.",
                stacklevel=2,
            )
            return [None for _ in self.remotes]
        return self._same_request(self.remotes, "render", None)

    def has_attr(self, attr_name: str) -> bool:
        remotes = tuple(self._get_target_remotes(indices=None))
        return all(self._same_request(remotes, "has_attr", attr_name))

    def get_attr(
        self,
        attr_name: str,
        indices: VecEnvIndices = None,
    ) -> list[Any]:
        remotes = tuple(self._get_target_remotes(indices))
        return self._same_request(remotes, "get_attr", attr_name)

    def set_attr(
        self,
        attr_name: str,
        value: Any,
        indices: VecEnvIndices = None,
    ) -> None:
        remotes = tuple(self._get_target_remotes(indices))
        self._same_request(remotes, "set_attr", (attr_name, value))

    def env_method(
        self,
        method_name: str,
        *method_args: object,
        indices: VecEnvIndices = None,
        **method_kwargs: object,
    ) -> list[Any]:
        remotes = tuple(self._get_target_remotes(indices))
        return self._same_request(
            remotes,
            "env_method",
            (method_name, method_args, method_kwargs),
        )

    def env_is_wrapped(
        self,
        wrapper_class: type[gym.Wrapper],
        indices: VecEnvIndices = None,
    ) -> list[bool]:
        remotes = tuple(self._get_target_remotes(indices))
        return self._same_request(remotes, "is_wrapped", wrapper_class)

    def close(self) -> None:
        if self.closed:
            return
        pending_step = self.waiting
        self.waiting = False
        self.closed = True
        failures = self._close_process_resources(send_close=not pending_step)
        failures.extend(self._close_resource_owner())
        if failures:
            raise BaseExceptionGroup(
                "failed to close subprocess rollout workers",
                failures,
            )

    def _same_request(
        self,
        remotes: tuple[Connection, ...],
        command: str,
        data: object,
    ) -> list[Any]:
        return self._request(
            remotes,
            tuple((command, data) for _ in remotes),
            command=command,
        )

    def _request(
        self,
        remotes: tuple[Connection, ...],
        requests: tuple[tuple[str, object], ...],
        *,
        command: str,
    ) -> list[Any]:
        self._send(remotes, requests, command=command)
        return self._receive(remotes, command=command)

    def _send(
        self,
        remotes: tuple[Connection, ...],
        requests: tuple[tuple[str, object], ...],
        *,
        command: str,
    ) -> None:
        if self.closed:
            raise RuntimeError("subprocess rollout environment is closed")
        if len(remotes) != len(requests):
            raise ValueError("rollout request count does not match remote count")
        for remote, request in zip(remotes, requests, strict=True):
            worker_index = self._worker_index(remote)
            try:
                remote.send(request)
            except (BrokenPipeError, EOFError, OSError) as error:
                self._abort_and_raise(
                    RolloutWorkerCommunicationError(
                        worker_index=worker_index,
                        command=command,
                        detail=f"{type(error).__name__}: {error}",
                    ),
                    cause=error,
                )

    def _receive(
        self,
        remotes: tuple[Connection, ...],
        *,
        command: str,
    ) -> list[Any]:
        deadline = time.monotonic() + self._response_timeout_seconds
        pending = {id(remote): remote for remote in remotes}
        results: dict[int, Any] = {}
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                self._abort_and_raise(
                    RolloutWorkerTimeoutError(
                        command=command,
                        pending_worker_indices=tuple(
                            self._worker_index(remote)
                            for remote in pending.values()
                        ),
                        timeout_seconds=self._response_timeout_seconds,
                    )
                )
            try:
                ready = wait(tuple(pending.values()), timeout=remaining)
            except (OSError, ValueError) as error:
                self._abort_and_raise(
                    RolloutWorkerCommunicationError(
                        worker_index=None,
                        command=command,
                        detail=f"{type(error).__name__}: {error}",
                    ),
                    cause=error,
                )
            if not ready:
                self._abort_and_raise(
                    RolloutWorkerTimeoutError(
                        command=command,
                        pending_worker_indices=tuple(
                            self._worker_index(remote)
                            for remote in pending.values()
                        ),
                        timeout_seconds=self._response_timeout_seconds,
                    )
                )
            for remote in ready:
                pending.pop(id(remote))
                worker_index = self._worker_index(remote)
                try:
                    payload = remote.recv()
                except (BrokenPipeError, EOFError, OSError) as error:
                    self._abort_and_raise(
                        RolloutWorkerCommunicationError(
                            worker_index=worker_index,
                            command=command,
                            detail=f"{type(error).__name__}: {error}",
                        ),
                        cause=error,
                    )
                if isinstance(payload, _WorkerFailurePayload):
                    self._abort_and_raise(
                        RolloutWorkerRemoteError(
                            worker_index=worker_index,
                            command=command,
                            failure=payload,
                        )
                    )
                results[worker_index] = payload
        return [results[self._worker_index(remote)] for remote in remotes]

    def _worker_index(self, remote: Connection) -> int:
        for index, candidate in enumerate(self.remotes):
            if candidate is remote:
                return index
        raise ValueError("remote does not belong to this rollout environment")

    def _abort_and_raise(
        self,
        error: RolloutWorkerError,
        *,
        cause: BaseException | None = None,
    ) -> None:
        if not self.closed:
            self.waiting = False
            self.closed = True
            failures = self._close_process_resources(send_close=False)
            failures.extend(self._close_resource_owner())
            self._annotate_cleanup_failures(error, failures)
        if cause is None:
            raise error from None
        raise error from cause

    @staticmethod
    def _annotate_cleanup_failures(
        error: BaseException,
        failures: list[BaseException],
    ) -> None:
        if failures:
            detail = "; ".join(
                f"{type(failure).__name__}: {failure}" for failure in failures
            )
            error.add_note(f"rollout cleanup failures: {detail}")

    def _close_resource_owner(self) -> list[BaseException]:
        owner = self._resource_owner
        self._resource_owner = None
        if owner is None:
            return []
        try:
            owner.close()
        except BaseException as error:
            return [error]
        return []

    def _close_process_resources(
        self,
        *,
        send_close: bool,
    ) -> list[BaseException]:
        failures: list[BaseException] = []
        remotes = tuple(getattr(self, "remotes", ()))
        processes = tuple(getattr(self, "processes", ()))
        process_errors = (OSError, ValueError, AssertionError)
        if send_close:
            for remote in remotes:
                try:
                    remote.send(("close", None))
                except (BrokenPipeError, EOFError, OSError) as error:
                    failures.append(error)
        else:
            for process in processes:
                try:
                    if process.is_alive():
                        process.terminate()
                except process_errors as error:
                    failures.append(error)

        def join_until(deadline: float) -> None:
            for process in processes:
                try:
                    process.join(timeout=max(0.0, deadline - time.monotonic()))
                except process_errors as error:
                    failures.append(error)

        join_until(time.monotonic() + ROLLOUT_CLOSE_PHASE_TIMEOUT_SECONDS)
        for process in processes:
            try:
                if process.is_alive():
                    process.terminate()
            except process_errors as error:
                failures.append(error)
        join_until(time.monotonic() + ROLLOUT_CLOSE_PHASE_TIMEOUT_SECONDS)
        for process in processes:
            try:
                if process.is_alive():
                    process.kill()
            except process_errors as error:
                failures.append(error)
        join_until(time.monotonic() + ROLLOUT_CLOSE_PHASE_TIMEOUT_SECONDS)

        for remote in (*remotes, *tuple(getattr(self, "work_remotes", ()))):
            try:
                remote.close()
            except OSError as error:
                failures.append(error)
        for process in processes:
            try:
                if process.is_alive():
                    failures.append(
                        RuntimeError(
                            f"rollout worker pid={process.pid} did not stop"
                        )
                    )
                else:
                    process.close()
            except process_errors as error:
                failures.append(error)
        return failures


@dataclass(frozen=True, slots=True)
class _LocalRolloutEnvFactory:
    episode: PreparedEpisode
    assignment: RolloutWorkerAssignment
    action_schema_payload: Mapping[str, object]
    normalizer_payload: Mapping[str, object]
    initial_cash: float
    random_window_min_transitions: int | None
    fees: FeeSchedule

    def __call__(self) -> WBRGymEnv:
        configure_single_thread_runtime()
        action_schema = ActionSchema.from_dict(self.action_schema_payload)
        return _WorkerWBRGymEnv(
            self.episode,
            worker_seed=self.assignment.seed,
            attachment=None,
            action_schema=action_schema,
            normalizer=TrainOnlyNormalizer.from_dict(self.normalizer_payload),
            initial_cash=self.initial_cash,
            include_critic_context=True,
            random_window_min_transitions=self.random_window_min_transitions,
            fees=self.fees,
        )


class RolloutEnvironment:
    """VecEnv plus the resources and immutable execution manifest it owns."""

    def __init__(
        self,
        *,
        vec_env: VecEnv,
        requested_backend: RolloutBackend,
        resolved_backend: ResolvedRolloutBackend,
        assignments: tuple[RolloutWorkerAssignment, ...],
        factories: tuple[object, ...],
        shared_owner: SharedPreparedEpisodeOwner | None,
        random_window_min_transitions: int | None,
    ) -> None:
        self.vec_env = vec_env
        self.requested_backend = requested_backend
        self.resolved_backend = resolved_backend
        self.assignments = assignments
        self.factories = factories
        self._shared_owner = shared_owner
        self.random_window_min_transitions = random_window_min_transitions
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def borrowed_shared_descriptor(
        self,
    ) -> SharedPreparedEpisodeDescriptor | None:
        """Borrow the rollout owner's descriptor without transferring ownership."""

        if self._closed:
            raise RuntimeError("cannot borrow from a closed rollout environment")
        if self._shared_owner is None:
            return None
        return self._shared_owner.descriptor

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": ROLLOUT_SCHEMA_VERSION,
            "requested_backend": self.requested_backend,
            "resolved_backend": self.resolved_backend,
            "vector_env_class": type(self.vec_env).__name__,
            "learner_processes": 1,
            "rollout_worker_processes": (
                len(self.assignments) if self.resolved_backend == "subproc" else 0
            ),
            "start_method": (
                ROLLOUT_START_METHOD
                if self.resolved_backend == "subproc"
                else None
            ),
            "data_transport": (
                "multiprocessing.shared_memory_read_only"
                if self.resolved_backend == "subproc"
                else "in_process_immutable_reference"
            ),
            "shared_memory_bytes": (
                0
                if self._shared_owner is None
                else self._shared_owner.descriptor.shared_memory_bytes
            ),
            "single_thread": {
                "environment": dict(SINGLE_THREAD_ENVIRONMENT),
                "torch_num_threads": 1,
                "torch_num_interop_threads": 1,
            },
            "episode_scope": ("full_training_period" if self.random_window_min_transitions is None
                              else "random_contiguous_train_window"),
            "window_start": ("training_start" if self.random_window_min_transitions is None
                             else "uniform_over_all_valid_training_starts"),
            "window_length": ("all_training_transitions" if self.random_window_min_transitions is None
                              else "uniform_from_minimum_to_remaining_transitions"),
            "minimum_window_transitions": self.random_window_min_transitions,
            "account_chains": "one_independent_candidate_account_per_env",
            "static_reference_in_rollout": False,
            "assignments": [item.as_dict() for item in self.assignments],
        }

    def close(self) -> None:
        if self._closed:
            return
        failures: list[BaseException] = []
        try:
            self.vec_env.close()
        except BaseException as error:
            failures.append(error)
        if self._shared_owner is not None:
            try:
                self._shared_owner.close()
            except BaseException as error:
                failures.append(error)
        self._closed = True
        if failures:
            raise BaseExceptionGroup(
                "failed to close PPO rollout environment",
                failures,
            )

    def __enter__(self) -> "RolloutEnvironment":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def build_rollout_env(
    episode: PreparedEpisode,
    action_schema: ActionSchema,
    normalizer: TrainOnlyNormalizer,
    *,
    initial_cash: float,
    assignments: tuple[RolloutWorkerAssignment, ...],
    backend: RolloutBackend,
    random_window_min_transitions: int | None,
    fees: FeeSchedule = DEFAULT_FEE_SCHEDULE,
) -> RolloutEnvironment:
    """Build the requested VecEnv and own every associated shared resource."""

    if not assignments:
        raise ValueError("rollout assignments must not be empty")
    expected_indices = tuple(range(len(assignments)))
    if tuple(item.worker_index for item in assignments) != expected_indices:
        raise ValueError("rollout worker indices must be contiguous and ordered")
    resolved = resolve_rollout_backend(backend, len(assignments))
    configure_single_thread_runtime()
    schema_payload = action_schema.to_dict()
    normalizer_payload = normalizer.to_dict()
    owner: SharedPreparedEpisodeOwner | None = None
    try:
        if resolved == "subproc":
            owner = SharedPreparedEpisodeOwner.create(episode)
            factories: tuple[object, ...] = tuple(
                RolloutEnvFactory(
                    episode_descriptor=owner.descriptor,
                    assignment=assignment,
                    action_schema_payload=schema_payload,
                    normalizer_payload=normalizer_payload,
                    initial_cash=float(initial_cash),
                    random_window_min_transitions=random_window_min_transitions,
                    fees=fees,
                )
                for assignment in assignments
            )
            vec_env: VecEnv = _DeterministicSubprocVecEnv(
                factories,  # type: ignore[arg-type]
                start_method=ROLLOUT_START_METHOD,
                resource_owner=owner,
            )
        else:
            factories = tuple(
                _LocalRolloutEnvFactory(
                    episode=episode,
                    assignment=assignment,
                    action_schema_payload=schema_payload,
                    normalizer_payload=normalizer_payload,
                    initial_cash=float(initial_cash),
                    random_window_min_transitions=random_window_min_transitions,
                    fees=fees,
                )
                for assignment in assignments
            )
            vec_env = DummyVecEnv(factories)  # type: ignore[arg-type]
        return RolloutEnvironment(
            vec_env=vec_env,
            requested_backend=backend,
            resolved_backend=resolved,
            assignments=assignments,
            factories=factories,
            shared_owner=owner,
            random_window_min_transitions=random_window_min_transitions,
        )
    except BaseException as build_error:
        if owner is not None:
            try:
                owner.close()
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "rollout environment build and cleanup both failed",
                    [build_error, cleanup_error],
                )
        raise


__all__ = [
    "ROLLOUT_SCHEMA_VERSION",
    "ROLLOUT_START_METHOD",
    "ROLLOUT_RESPONSE_TIMEOUT_SECONDS",
    "RolloutBackend",
    "RolloutEnvFactory",
    "RolloutEnvironment",
    "RolloutWorkerCommunicationError",
    "RolloutWorkerError",
    "RolloutWorkerProtocolError",
    "RolloutWorkerRemoteError",
    "RolloutWorkerTimeoutError",
    "RolloutWorkerAssignment",
    "SINGLE_THREAD_ENVIRONMENT",
    "SharedArrayDescriptor",
    "build_rollout_env",
    "build_worker_assignments",
    "configure_single_thread_runtime",
    "resolve_rollout_backend",
]
