"""Causal net-performance metrics and the PPO dense episode reward."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


ANNUALIZATION_DAYS = 252
TRADING_DAYS_PER_YEAR = float(ANNUALIZATION_DAYS)
CALMAR_DRAWDOWN_EPSILON = 1e-6
MAX_DRAWDOWN_PENALTY_WEIGHT = 1.0
REWARD_SCHEMA_VERSION = "horizon-balanced-return-incremental-max-drawdown-v4"

@dataclass(frozen=True)
class PerformanceMetrics:
    transition_count: int
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
    calmar: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "transition_count": self.transition_count,
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "annualized_volatility": self.annualized_volatility,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "calmar": self.calmar,
        }


@dataclass(frozen=True)
class StreamingPerformanceState:
    """Sufficient statistics for one causal, already-net episode return stream."""

    horizon_transitions: int
    observed_transitions: int = 0
    cumulative_log_return: float = 0.0
    simple_return_sum: float = 0.0
    simple_return_square_sum: float = 0.0
    peak_log_equity: float = 0.0
    max_drawdown: float = 0.0

    def __post_init__(self) -> None:
        if type(self.horizon_transitions) is not int or self.horizon_transitions <= 0:
            raise ValueError("horizon_transitions must be a positive integer")
        if (
            type(self.observed_transitions) is not int
            or not 0 <= self.observed_transitions <= self.horizon_transitions
        ):
            raise ValueError("observed_transitions must be inside the fixed horizon")
        values = (
            self.cumulative_log_return,
            self.simple_return_sum,
            self.simple_return_square_sum,
            self.peak_log_equity,
            self.max_drawdown,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("streaming performance state must be finite")
        if self.simple_return_square_sum < 0.0:
            raise ValueError("simple return square sum must be non-negative")
        if self.peak_log_equity < max(0.0, self.cumulative_log_return):
            raise ValueError("peak log equity cannot be below current log equity")
        if not 0.0 <= self.max_drawdown <= 1.0:
            raise ValueError("max_drawdown must be in [0, 1]")

    @classmethod
    def initial(cls, horizon_transitions: int) -> "StreamingPerformanceState":
        return cls(horizon_transitions=horizon_transitions)

    @property
    def is_complete(self) -> bool:
        return self.observed_transitions == self.horizon_transitions

    def advance(self, net_log_return: float) -> "StreamingPerformanceState":
        if self.is_complete:
            raise ValueError("streaming performance state reached its fixed horizon")
        value = float(net_log_return)
        if not math.isfinite(value):
            raise ValueError("net_log_return must be finite")
        simple_return = math.expm1(value)
        cumulative = self.cumulative_log_return + value
        simple_sum = self.simple_return_sum + simple_return
        simple_square_sum = self.simple_return_square_sum + simple_return**2
        if not all(
            math.isfinite(item)
            for item in (simple_return, cumulative, simple_sum, simple_square_sum)
        ):
            raise ValueError("streaming performance statistics overflowed")
        peak = max(self.peak_log_equity, cumulative)
        drawdown = -math.expm1(cumulative - peak)
        return StreamingPerformanceState(
            horizon_transitions=self.horizon_transitions,
            observed_transitions=self.observed_transitions + 1,
            cumulative_log_return=cumulative,
            simple_return_sum=simple_sum,
            simple_return_square_sum=simple_square_sum,
            peak_log_equity=peak,
            max_drawdown=max(self.max_drawdown, drawdown),
        )

    def performance(self) -> PerformanceMetrics:
        return self._performance

    @cached_property
    def _performance(self) -> PerformanceMetrics:
        count = self.horizon_transitions
        total_return = math.expm1(self.cumulative_log_return)
        annualized_return = math.expm1(
            self.cumulative_log_return * TRADING_DAYS_PER_YEAR / count
        )
        mean = self.simple_return_sum / count
        if count > 1:
            centered = (
                self.simple_return_square_sum
                - self.simple_return_sum * self.simple_return_sum / count
            )
            daily_std = math.sqrt(max(0.0, centered / (count - 1)))
        else:
            daily_std = 0.0
        annualized_volatility = daily_std * math.sqrt(TRADING_DAYS_PER_YEAR)
        sharpe = (
            mean / daily_std * math.sqrt(TRADING_DAYS_PER_YEAR)
            if daily_std > 0.0
            else 0.0
        )
        calmar = annualized_return / max(
            self.max_drawdown,
            CALMAR_DRAWDOWN_EPSILON,
        )
        values = (
            total_return,
            annualized_return,
            annualized_volatility,
            sharpe,
            calmar,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("performance metrics overflowed")
        return PerformanceMetrics(
            transition_count=count,
            total_return=total_return,
            annualized_return=annualized_return,
            annualized_volatility=annualized_volatility,
            sharpe=sharpe,
            max_drawdown=self.max_drawdown,
            calmar=calmar,
        )


@dataclass(frozen=True)
class EpisodeReward:
    net_log_return: float
    annualized_return_increment: float
    drawdown_increment_penalty: float
    horizon_scale: float
    reward: float
    metrics: PerformanceMetrics

    def as_dict(self) -> dict[str, object]:
        return {
            "net_log_return": self.net_log_return,
            "annualized_return_increment": self.annualized_return_increment,
            "drawdown_increment_penalty": self.drawdown_increment_penalty,
            "horizon_scale": self.horizon_scale,
            "reward": self.reward,
            "calmar": self.metrics.calmar,
            "annualized_return": self.metrics.annualized_return,
            "max_drawdown": self.metrics.max_drawdown,
        }


@dataclass(frozen=True)
class EpisodeRewardState:
    performance_state: StreamingPerformanceState

    @classmethod
    def initial(cls, horizon_transitions: int) -> "EpisodeRewardState":
        return cls(StreamingPerformanceState.initial(horizon_transitions))

    @property
    def is_complete(self) -> bool:
        return self.performance_state.is_complete

    def advance(
        self,
        net_log_return: float,
    ) -> tuple["EpisodeRewardState", EpisodeReward]:
        previous_max_drawdown = self.performance_state.max_drawdown
        previous_annualized_return = self.performance_state.performance().annualized_return
        state = self.performance_state.advance(net_log_return)
        metrics = state.performance()
        annualized_return_increment = (
            metrics.annualized_return - previous_annualized_return
        )
        drawdown_increment_penalty = MAX_DRAWDOWN_PENALTY_WEIGHT * (
            metrics.max_drawdown - previous_max_drawdown
        )
        horizon_scale = state.horizon_transitions / TRADING_DAYS_PER_YEAR
        reward = EpisodeReward(
            net_log_return=float(net_log_return),
            annualized_return_increment=annualized_return_increment,
            drawdown_increment_penalty=drawdown_increment_penalty,
            horizon_scale=horizon_scale,
            reward=horizon_scale * (annualized_return_increment - drawdown_increment_penalty),
            metrics=metrics,
        )
        return EpisodeRewardState(state), reward


def performance_from_log_rewards(
    log_rewards: Sequence[float] | NDArray[np.floating],
) -> PerformanceMetrics:
    rewards = np.asarray(log_rewards, dtype=np.float64)
    if rewards.ndim != 1 or not len(rewards):
        raise ValueError("log_rewards must be a non-empty vector")
    if not np.isfinite(rewards).all():
        raise ValueError("log_rewards must be finite")
    state = StreamingPerformanceState.initial(len(rewards))
    for reward in rewards:
        state = state.advance(float(reward))
    return state.performance()


def summarize_event_forward_returns(
    events: NDArray[np.bool_],
    gross_returns: NDArray[np.floating],
    *,
    horizons: tuple[int, ...],
) -> dict[str, dict[str, float | int | None]]:
    """Describe event returns without inventing an executable portfolio.

    ``events[T]`` is known at entry T; ``gross_returns[T]`` is the economic
    T-to-T+1 multiplier supplied by the settlement owner. Windows must fit
    entirely inside these arrays. Invalid price chains are excluded and
    counted, never zero-filled. Both event and signal-day weighting are shown.
    """
    events = np.asarray(events)
    gross = np.asarray(gross_returns)
    if events.dtype != np.bool_ or events.ndim != 2 or gross.shape != (len(events) - 1, events.shape[1]):
        raise ValueError("events must be boolean dates-by-stocks and gross_returns must have one fewer date")
    if not horizons or any(type(h) is not int or h <= 0 for h in horizons) or len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be distinct positive integers")
    output = {}
    for horizon in horizons:
        rows = max(0, len(events) - horizon)
        selected = events[:rows]
        accumulated = np.ones((rows, events.shape[1]), dtype=np.float64)
        for offset in range(horizon if rows else 0):
            piece = gross[offset:offset + rows]
            accumulated *= np.where(np.isfinite(piece) & (piece > 0), piece, np.nan)
        valid = selected & np.isfinite(accumulated)
        returns = accumulated - 1.0
        observed = returns[valid]
        counts = valid.sum(axis=1)
        active_days = counts > 0
        daily_sum = np.where(valid, returns, 0.0).sum(axis=1)
        output[str(horizon)] = {
            "signal_count": int(events.sum()),
            "right_censored_count": int(events[rows:].sum()),
            "eligible_event_count": int(selected.sum()),
            "missing_price_chain_count": int(selected.sum() - valid.sum()),
            "observed_event_count": int(valid.sum()),
            "signal_day_count": int(active_days.sum()),
            "mean_return": float(observed.mean()) if observed.size else None,
            "win_rate": float(np.mean(observed > 0)) if observed.size else None,
            "signal_day_equal_weight_mean_return": float(np.mean(daily_sum[active_days] / counts[active_days])) if active_days.any() else None,
        }
    return output


__all__ = [
    "ANNUALIZATION_DAYS",
    "CALMAR_DRAWDOWN_EPSILON",
    "MAX_DRAWDOWN_PENALTY_WEIGHT",
    "EpisodeReward",
    "EpisodeRewardState",
    "PerformanceMetrics",
    "REWARD_SCHEMA_VERSION",
    "StreamingPerformanceState",
    "performance_from_log_rewards",
    "summarize_event_forward_returns",
]
