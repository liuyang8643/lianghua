"""Torch-free rollout worker process for PPO subprocess environments.

Every spawned rollout worker imports only this module (plus ``env``) when it
unpickles its environment factory and runs the command loop.  Importing Torch
and Stable-Baselines3 inside each of 20+ workers used to commit ~1 GiB per
process for CUDA/DLL mappings and pushed the machine over its commit limit
(``WinError 1455`` while loading ``torch\\lib\\shm.dll``, 2026-09-21).  Account
simulation is pure NumPy/Numba, so the workers never need Torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.connection import Connection
import os
import traceback
from typing import Any, Mapping

SINGLE_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
for _thread_environment_name, _thread_environment_value in SINGLE_THREAD_ENVIRONMENT.items():
    os.environ[_thread_environment_name] = _thread_environment_value
del _thread_environment_name, _thread_environment_value

import cloudpickle
import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from env.action_schema import ActionSchema
from env.backtest import PreparedEpisode
from env.encoder import TrainOnlyNormalizer
from env.fees import FeeSchedule
from env.gym_adapter import WBRGymEnv
from env.shared_episode import AttachedPreparedEpisode, SharedPreparedEpisodeDescriptor


def configure_worker_single_thread() -> None:
    """Pin native math libraries to one thread; Torch is never imported here."""

    for name, value in SINGLE_THREAD_ENVIRONMENT.items():
        os.environ[name] = value


@dataclass(frozen=True, slots=True)
class RolloutWorkerAssignment:
    """One stable mapping from a vector slot to its independent RNG seed."""

    worker_index: int
    seed: int

    def __post_init__(self) -> None:
        if self.worker_index < 0:
            raise ValueError("worker index must be non-negative")
        if self.seed < 0:
            raise ValueError("worker seed must be non-negative")

    def as_dict(self) -> dict[str, object]:
        return {
            "worker_index": self.worker_index,
            "seed": self.seed,
        }


def build_worker_assignments(
    *,
    n_envs: int,
    base_seed: int,
) -> tuple[RolloutWorkerAssignment, ...]:
    """Build deterministic independent full-period vector slots."""

    if type(n_envs) is not int or n_envs <= 0:
        raise ValueError("n_envs must be a positive int")
    if type(base_seed) is not int or base_seed < 0:
        raise ValueError("base_seed must be a non-negative int")
    return tuple(
        RolloutWorkerAssignment(worker_index=index, seed=base_seed + index)
        for index in range(n_envs)
    )


class WorkerWBRGymEnv(WBRGymEnv):
    """WBR environment with one sealed initial seed and optional attachment."""

    def __init__(
        self,
        episode: PreparedEpisode,
        *,
        worker_seed: int,
        attachment: AttachedPreparedEpisode | None,
        **kwargs: object,
    ) -> None:
        super().__init__(episode, **kwargs)
        self.worker_seed = int(worker_seed)
        self._first_reset = True
        self._attachment = attachment
        self.action_space.seed(self.worker_seed)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, object]]:
        if self._first_reset:
            effective_seed = self.worker_seed if seed is None else int(seed)
            if effective_seed != self.worker_seed:
                raise ValueError(
                    "first rollout reset seed differs from the sealed worker assignment"
                )
            self._first_reset = False
            seed = effective_seed
        return super().reset(seed=seed, options=options)

    def close(self) -> None:
        attachment = self._attachment
        self._attachment = None
        try:
            super().close()
        finally:
            if attachment is not None:
                # Drop the environment's last episode reference before closing
                # its SharedMemory mappings.
                self.episode = None  # type: ignore[assignment]
                attachment.close()


@dataclass(frozen=True, slots=True)
class RolloutEnvFactory:
    """Top-level spawn-pickleable factory containing no PreparedEpisode/path."""

    episode_descriptor: SharedPreparedEpisodeDescriptor
    assignment: RolloutWorkerAssignment
    action_schema_payload: Mapping[str, object]
    normalizer_payload: Mapping[str, object]
    initial_cash: float
    random_window_min_transitions: int | None
    fees: FeeSchedule

    def __call__(self) -> WBRGymEnv:
        configure_worker_single_thread()
        attached = self.episode_descriptor.attach()
        try:
            action_schema = ActionSchema.from_dict(self.action_schema_payload)
            normalizer = TrainOnlyNormalizer.from_dict(self.normalizer_payload)
            return WorkerWBRGymEnv(
                attached.episode,
                worker_seed=self.assignment.seed,
                attachment=attached,
                action_schema=action_schema,
                normalizer=normalizer,
                initial_cash=self.initial_cash,
                include_critic_context=True,
                random_window_min_transitions=self.random_window_min_transitions,
                fees=self.fees,
            )
        except BaseException as construction_error:
            try:
                attached.close()
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "rollout worker construction and cleanup both failed",
                    [construction_error, cleanup_error],
                )
            raise


class CloudpickleWrapper:
    """Pickle an environment factory with cloudpickle (mirrors SB3 without importing it)."""

    def __init__(self, var: Any) -> None:
        self.var = var

    def __getstate__(self) -> bytes:
        return cloudpickle.dumps(self.var)

    def __setstate__(self, state: bytes) -> None:
        self.var = cloudpickle.loads(state)


@dataclass(frozen=True, slots=True)
class WorkerFailurePayload:
    exception_type: str
    message: str
    traceback: str


def _is_wrapped(env: gym.Env, wrapper_class: type) -> bool:
    current: Any = env
    while current is not None:
        if isinstance(current, wrapper_class):
            return True
        current = getattr(current, "env", None)
    return False


def _command_loop(remote: Connection, parent_remote: Connection, env_fn_wrapper: CloudpickleWrapper) -> None:
    """Stable-Baselines3 SubprocVecEnv worker protocol, reimplemented without Torch."""

    parent_remote.close()
    env: gym.Env = env_fn_wrapper.var()
    reset_info: dict[str, Any] | None = {}
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, terminated, truncated, info = env.step(data)
                done = terminated or truncated
                info["TimeLimit.truncated"] = truncated and not terminated
                if done:
                    info["terminal_observation"] = observation
                    observation, reset_info = env.reset()
                remote.send((observation, reward, done, info, reset_info))
            elif cmd == "reset":
                maybe_options = {"options": data[1]} if data[1] else {}
                observation, reset_info = env.reset(seed=data[0], **maybe_options)
                remote.send((observation, reset_info))
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = env.get_wrapper_attr(data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(env.get_wrapper_attr(data))
            elif cmd == "has_attr":
                try:
                    env.get_wrapper_attr(data)
                    remote.send(True)
                except AttributeError:
                    remote.send(False)
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))  # type: ignore[func-returns-value]
            elif cmd == "is_wrapped":
                remote.send(_is_wrapped(env, data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break
        except KeyboardInterrupt:
            break


def exception_reporting_worker(
    remote: Connection,
    parent_remote: Connection,
    env_fn_wrapper: CloudpickleWrapper,
) -> None:
    """Run the command loop while preserving any uncaught exception as text."""

    try:
        _command_loop(remote, parent_remote, env_fn_wrapper)
    except BaseException as error:
        failure = WorkerFailurePayload(
            exception_type=f"{type(error).__module__}.{type(error).__qualname__}",
            message=str(error),
            traceback=traceback.format_exc(),
        )
        try:
            remote.send(failure)
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        for connection in (remote, parent_remote):
            try:
                connection.close()
            except OSError:
                pass


__all__ = [
    "CloudpickleWrapper", "RolloutEnvFactory", "RolloutWorkerAssignment", "SINGLE_THREAD_ENVIRONMENT",
    "WorkerFailurePayload", "WorkerWBRGymEnv", "build_worker_assignments",
    "configure_worker_single_thread", "exception_reporting_worker",
]
