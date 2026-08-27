"""Deterministic policy rollouts through the canonical Gym/domain path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from env.gym_adapter import WBRGymEnv
from env.metrics import PerformanceMetrics, performance_from_log_rewards


ActionProvider = Callable[[NDArray[np.float32]], NDArray[np.float32]]


@dataclass(frozen=True)
class RolloutTrace:
    decision_dates: tuple[str, ...]
    next_decision_dates: tuple[str, ...]
    rewards: NDArray[np.float64]
    portfolio_returns: NDArray[np.float64]
    nav: NDArray[np.float64]
    exposure: NDArray[np.float64]
    actions: NDArray[np.float32]
    day_configs: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        size = len(self.rewards)
        if not (
            len(self.decision_dates)
            == len(self.next_decision_dates)
            == len(self.portfolio_returns)
            == len(self.exposure)
            == len(self.actions)
            == len(self.day_configs)
            == size
        ):
            raise ValueError("rollout transition fields have inconsistent lengths")
        if self.nav.shape != (size + 1,):
            raise ValueError("rollout nav must contain the initial point and every settlement")
        for values in (
            self.rewards,
            self.portfolio_returns,
            self.nav,
            self.exposure,
            self.actions,
        ):
            if not np.isfinite(values).all():
                raise ValueError("rollout values must be finite")

    @property
    def metrics(self) -> PerformanceMetrics:
        return performance_from_log_rewards(self.rewards)

    @property
    def average_exposure(self) -> float:
        return float(self.exposure.mean())

    def as_summary(self) -> dict[str, object]:
        return {
            "start": self.decision_dates[0],
            "end": self.next_decision_dates[-1],
            "average_exposure": self.average_exposure,
            "metrics": self.metrics.as_dict(),
        }


def run_episode(
    env: WBRGymEnv,
    action_provider: ActionProvider,
    *,
    seed: int = 0,
) -> RolloutTrace:
    observation, reset_info = env.reset(seed=seed)
    initial_nav = float(reset_info["nav"])
    decision_dates: list[str] = []
    next_dates: list[str] = []
    rewards: list[float] = []
    returns: list[float] = []
    nav = [initial_nav]
    exposure: list[float] = []
    actions: list[NDArray[np.float32]] = []
    configs: list[Mapping[str, object]] = []

    terminated = False
    while not terminated:
        action = np.asarray(action_provider(observation.copy()), dtype=np.float32)
        observation, reward, terminated, truncated, info = env.step(action)
        if truncated:
            raise RuntimeError("sealed market-data episodes must not truncate")
        decision_dates.append(str(info["decision_date"]))
        next_dates.append(str(info["next_decision_date"]))
        rewards.append(float(reward))
        returns.append(float(info["portfolio_return"]))
        nav.append(float(info["nav"]))
        exposure.append(float(info["exposure"]))
        actions.append(action.copy())
        configs.append(dict(info["day_config"]))

    if len(rewards) != env.episode.transition_count:
        raise RuntimeError("rollout did not consume the exact sealed transition count")
    return RolloutTrace(
        decision_dates=tuple(decision_dates),
        next_decision_dates=tuple(next_dates),
        rewards=np.asarray(rewards, dtype=np.float64),
        portfolio_returns=np.asarray(returns, dtype=np.float64),
        nav=np.asarray(nav, dtype=np.float64),
        exposure=np.asarray(exposure, dtype=np.float64),
        actions=np.stack(actions).astype(np.float32),
        day_configs=tuple(configs),
    )


def dynamic_config_summary(trace: RolloutTrace) -> dict[str, object]:
    """Describe actual continuous, binary and discrete variation in a rollout."""

    configs = trace.day_configs
    continuous: dict[str, list[float]] = {}
    categorical: dict[str, list[object]] = {}
    factor_names = tuple(configs[0]["weights"])
    filter_names = tuple(configs[0]["filter_factors"])
    for name in factor_names:
        continuous[f"factor_weight.{name}"] = [
            float(config["weights"][name]) for config in configs
        ]
        categorical[f"factor_enabled.{name}"] = [
            bool(config["factor_enabled"][name]) for config in configs
        ]
    for name in filter_names:
        categorical[f"filter_flag.{name}"] = [
            bool(config["filter_factors"][name]) for config in configs
        ]
    for name in ("target_exposure", "rebalance_band_pct"):
        source_name = "cash_reserve_ratio" if name == "target_exposure" else name
        values = [float(config[source_name]) for config in configs]
        continuous[name] = [1.0 - value for value in values] if name == "target_exposure" else values
    for name in (
        "buy_n",
        "sell_m",
        "rebalance_now",
        "rebalance",
        "limit_up_protection",
    ):
        categorical[name] = [config[name] for config in configs]

    continuous_summary = {
        name: {
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
        for name, values in continuous.items()
    }
    categorical_summary: dict[str, object] = {}
    for name, values in categorical.items():
        counts: dict[str, int] = {}
        for value in values:
            key = str(value)
            counts[key] = counts.get(key, 0) + 1
        categorical_summary[name] = {
            "unique_count": len(counts),
            "counts": counts,
            "coverage": {key: count / len(values) for key, count in counts.items()},
        }
    return {
        "transition_count": len(configs),
        "continuous": continuous_summary,
        "categorical": categorical_summary,
    }


__all__ = [
    "ActionProvider",
    "RolloutTrace",
    "dynamic_config_summary",
    "run_episode",
]
