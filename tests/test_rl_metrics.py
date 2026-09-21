import math

import numpy as np
import pytest

from env.metrics import (
    CALMAR_DRAWDOWN_EPSILON,
    MAX_DRAWDOWN_PENALTY_WEIGHT,
    EpisodeRewardState,
    REWARD_SCHEMA_VERSION,
    StreamingPerformanceState,
    performance_from_log_rewards,
)


def test_reward_is_positive_horizon_scaling_of_annualized_log_objective():
    returns = np.asarray((0.02, -0.01, 0.03, -0.005), dtype=np.float64)
    horizon = len(returns)
    state = EpisodeRewardState.initial(horizon)
    rewards = []
    previous_max_drawdown = 0.0
    for value in returns:
        log_return = float(np.log1p(value))
        state, reward = state.advance(log_return)
        rewards.append(reward.reward)
        # Every day carries the same weight per unit of log return (stationary in t).
        assert reward.annualized_log_return_increment == pytest.approx(log_return * 252 / horizon)
        assert reward.drawdown_increment_penalty == pytest.approx(
            MAX_DRAWDOWN_PENALTY_WEIGHT
            * (reward.metrics.max_drawdown - previous_max_drawdown)
        )
        assert reward.reward == pytest.approx(
            log_return - horizon / 252 * reward.drawdown_increment_penalty
        )
        previous_max_drawdown = reward.metrics.max_drawdown

    final_metrics = performance_from_log_rewards(np.log1p(returns))
    assert sum(rewards) == pytest.approx(
        horizon / 252 * (math.log1p(final_metrics.annualized_return) - final_metrics.max_drawdown)
    )
    assert state.is_complete
    assert (
        REWARD_SCHEMA_VERSION
        == "horizon-balanced-log-return-incremental-max-drawdown-v5"
    )


def test_log_return_reward_does_not_weight_late_days_by_compounded_equity():
    horizon = 40
    state = EpisodeRewardState.initial(horizon)
    for _ in range(horizon - 1):
        state, _ = state.advance(0.01)  # strong monotone growth, no drawdown
    _, early = EpisodeRewardState.initial(horizon).advance(0.005)
    _, late = state.advance(0.005)
    assert late.reward == pytest.approx(early.reward)


def test_prefix_annualization_zero_pads_unobserved_tail():
    state = StreamingPerformanceState.initial(252).advance(math.log1p(0.10))
    metrics = state.performance()

    assert metrics.total_return == pytest.approx(0.10)
    assert metrics.annualized_return == pytest.approx(0.10)
    assert metrics.transition_count == 252


def test_reward_compensates_horizon_scale_without_changing_metrics():
    rewards = []
    for horizon in (20, 252, 2520):
        _, reward = EpisodeRewardState.initial(horizon).advance(1e-5)
        rewards.append(reward.reward)
        assert reward.horizon_scale == pytest.approx(horizon / 252)
        assert reward.metrics.total_return == pytest.approx(math.expm1(1e-5))
        assert reward.reward == pytest.approx(1e-5)
    assert max(rewards) / min(rewards) < 1.0001


def test_calmar_is_finite_with_explicit_zero_drawdown_convention():
    metrics = performance_from_log_rewards(np.log1p(np.full(10, 0.001)))

    assert metrics.max_drawdown == 0.0
    assert metrics.calmar == pytest.approx(
        metrics.annualized_return / CALMAR_DRAWDOWN_EPSILON
    )
    assert math.isfinite(metrics.calmar)


def test_reward_state_rejects_non_finite_net_returns():
    state = EpisodeRewardState.initial(2)

    with pytest.raises(ValueError, match="finite"):
        state.advance(float("nan"))
