"""Thin Gymnasium action adapter over the canonical episode session."""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray

from env.action_schema import ActionSchema
from env.backtest import EpisodeSession, PreparedEpisode
from env.contracts import AccountState, PolicyMemory
from env.encoder import TrainOnlyNormalizer
from env.fees import DEFAULT_FEE_SCHEDULE, FeeSchedule


class WBRGymEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """Gym API plus Box action decoding; domain state lives in EpisodeSession."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        episode: PreparedEpisode,
        *,
        action_schema: ActionSchema | None = None,
        normalizer: TrainOnlyNormalizer | None = None,
        initial_cash: float = 1_000_000.0,
        diagnostics: str = "minimal",
        include_critic_context: bool = False,
        random_window_min_transitions: int | None = None,
        fees: FeeSchedule = DEFAULT_FEE_SCHEDULE,
    ) -> None:
        super().__init__()
        self.session = EpisodeSession(
            episode,
            action_schema=action_schema,
            normalizer=normalizer,
            initial_cash=initial_cash,
            diagnostics=diagnostics,
            include_critic_context=include_critic_context,
            fees=fees,
        )
        self.episode = self.session.episode
        self.action_schema = self.session.action_schema
        self.normalizer = self.session.normalizer
        self.initial_cash = self.session.initial_cash
        self.include_critic_context = self.session.include_critic_context
        if random_window_min_transitions is not None and (
            type(random_window_min_transitions) is not int
            or not 1 <= random_window_min_transitions <= episode.transition_count
        ):
            raise ValueError("random window minimum must fit the sealed episode")
        self.random_window_min_transitions = random_window_min_transitions
        low, high = self.action_schema.space_bounds
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        bound = np.finfo(np.float32).max
        self.observation_space = spaces.Box(
            low=-bound,
            high=bound,
            shape=(self.session.observation_dimension,),
            dtype=np.float32,
        )

    @property
    def current_account(self) -> AccountState:
        return self.session.current_account

    @property
    def current_index(self) -> int:
        return self.session.current_index

    @property
    def current_policy_memory(self) -> PolicyMemory:
        return self.session.current_policy_memory

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, object]]:
        super().reset(seed=seed)
        if options:
            unknown = set(options) - {"account", "policy_memory"}
            if unknown:
                raise ValueError(f"unsupported reset options: {sorted(unknown)}")
        has_account = options is not None and "account" in options
        has_policy_memory = options is not None and "policy_memory" in options
        if has_account != has_policy_memory:
            raise ValueError(
                "reset options 'account' and 'policy_memory' must be supplied together"
            )
        account = options["account"] if has_account else None
        policy_memory = options["policy_memory"] if has_policy_memory else None
        if account is not None and not isinstance(account, AccountState):
            raise TypeError("reset option 'account' must be AccountState")
        if policy_memory is not None and not isinstance(policy_memory, PolicyMemory):
            raise TypeError("reset option 'policy_memory' must be PolicyMemory")
        decision_start: int | None = None
        decision_stop: int | None = None
        if self.random_window_min_transitions is not None:
            minimum = self.random_window_min_transitions
            latest_start_exclusive = self.episode.decision_stop - minimum
            if latest_start_exclusive == self.episode.decision_start:
                decision_start = self.episode.decision_start
            else:
                decision_start = int(
                    self.np_random.integers(
                        self.episode.decision_start,
                        latest_start_exclusive,
                    )
                )
            maximum = self.episode.decision_stop - decision_start - 1
            transitions = int(self.np_random.integers(minimum, maximum + 1))
            decision_stop = decision_start + transitions + 1
        observation, info = self.session.reset(
            account=account,
            policy_memory=policy_memory,
            decision_start=decision_start,
            decision_stop=decision_stop,
        )
        if self.random_window_min_transitions is not None:
            info["random_window"] = True
        return observation, info

    def step(
        self,
        action: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, object]]:
        raw_action = np.asarray(action, dtype=np.float32)
        config = self.action_schema.decode(raw_action)
        transition = self.session.step(config)
        info = dict(transition.info)
        info["raw_action"] = raw_action.copy()
        return (
            transition.observation,
            float(transition.step_result.reward),
            transition.step_result.terminated,
            False,
            info,
        )

    def render(self) -> None:
        return None


__all__ = ["WBRGymEnv"]
