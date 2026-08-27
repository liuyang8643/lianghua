"""Metrics for next-open reward traces produced by :mod:`env`."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray


TRADING_DAYS_PER_YEAR = 252.0


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


def equity_from_log_rewards(
    log_rewards: Sequence[float] | NDArray[np.floating],
) -> NDArray[np.float64]:
    rewards = np.asarray(log_rewards, dtype=np.float64)
    if rewards.ndim != 1 or not len(rewards):
        raise ValueError("log_rewards must be a non-empty vector")
    if not np.isfinite(rewards).all():
        raise ValueError("log_rewards must be finite")
    return np.concatenate((np.ones(1), np.exp(np.cumsum(rewards))))


def performance_from_log_rewards(
    log_rewards: Sequence[float] | NDArray[np.floating],
) -> PerformanceMetrics:
    rewards = np.asarray(log_rewards, dtype=np.float64)
    equity = equity_from_log_rewards(rewards)
    simple_returns = np.expm1(rewards)
    total_return = float(equity[-1] - 1.0)
    annualized_return = float(
        math.exp(float(rewards.sum()) * TRADING_DAYS_PER_YEAR / len(rewards)) - 1.0
    )
    daily_std = float(simple_returns.std(ddof=1)) if len(simple_returns) > 1 else 0.0
    annualized_volatility = daily_std * math.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (
        float(simple_returns.mean()) / daily_std * math.sqrt(TRADING_DAYS_PER_YEAR)
        if daily_std > 0.0
        else 0.0
    )
    peaks = np.maximum.accumulate(equity)
    drawdowns = 1.0 - equity / peaks
    max_drawdown = float(drawdowns.max())
    if max_drawdown > 0.0:
        calmar = annualized_return / max_drawdown
    elif annualized_return == 0.0:
        calmar = 0.0
    else:
        calmar = math.copysign(math.inf, annualized_return)
    return PerformanceMetrics(
        transition_count=len(rewards),
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        calmar=calmar,
    )


def interval_rewards(
    log_rewards: Sequence[float] | NDArray[np.floating],
    decision_dates: Sequence[str],
    next_decision_dates: Sequence[str],
    start: str,
    end: str,
) -> NDArray[np.float64]:
    rewards = np.asarray(log_rewards, dtype=np.float64)
    if not (
        len(rewards) == len(decision_dates) == len(next_decision_dates)
    ):
        raise ValueError("reward and date vectors must have equal lengths")
    start_date = np.datetime64(start, "D")
    end_date = np.datetime64(end, "D")
    if start_date > end_date:
        raise ValueError("interval start must not be later than end")
    left = np.asarray(decision_dates, dtype="datetime64[D]")
    right = np.asarray(next_decision_dates, dtype="datetime64[D]")
    selected = (left >= start_date) & (right <= end_date)
    result = rewards[selected]
    if not len(result):
        raise ValueError(f"interval contains no complete transitions: {start}..{end}")
    return result


def robust_calmar(
    log_rewards: Sequence[float] | NDArray[np.floating],
    decision_dates: Sequence[str],
    next_decision_dates: Sequence[str],
    folds: Iterable[tuple[str, str]],
) -> dict[str, object]:
    full = performance_from_log_rewards(log_rewards)
    fold_metrics = tuple(
        performance_from_log_rewards(
            interval_rewards(
                log_rewards,
                decision_dates,
                next_decision_dates,
                start,
                end,
            )
        )
        for start, end in folds
    )
    if not fold_metrics:
        raise ValueError("robust_calmar requires at least one fold")
    score = 0.5 * full.calmar + 0.5 * min(item.calmar for item in fold_metrics)
    return {
        "robust_calmar": float(score),
        "full": full.as_dict(),
        "folds": [item.as_dict() for item in fold_metrics],
    }


__all__ = [
    "PerformanceMetrics",
    "equity_from_log_rewards",
    "interval_rewards",
    "performance_from_log_rewards",
    "robust_calmar",
]
