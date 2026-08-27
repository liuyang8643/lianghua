import math

import numpy as np
import pytest

from env.metrics import (
    interval_rewards,
    performance_from_log_rewards,
    robust_calmar,
)


def test_metrics_use_compounded_open_to_open_log_rewards():
    rewards = np.log1p(np.asarray((0.10, -0.05, 0.02)))
    metrics = performance_from_log_rewards(rewards)

    assert metrics.total_return == pytest.approx(1.10 * 0.95 * 1.02 - 1.0)
    assert metrics.max_drawdown == pytest.approx(0.05)
    assert metrics.transition_count == 3
    assert math.isfinite(metrics.sharpe)


def test_interval_requires_both_ends_inside_the_fold():
    rewards = np.asarray((0.1, 0.2, 0.3))
    decision = ("2020-12-31", "2021-01-01", "2021-01-02")
    following = ("2021-01-01", "2021-01-02", "2021-01-03")

    selected = interval_rewards(
        rewards,
        decision,
        following,
        "2021-01-01",
        "2021-01-02",
    )
    np.testing.assert_array_equal(selected, np.asarray((0.2,)))


def test_robust_calmar_reports_full_and_each_sealed_fold():
    simple = np.asarray((0.02, -0.01, 0.03, -0.02, 0.01, 0.01))
    rewards = np.log1p(simple)
    decision = tuple(f"2020-01-0{day}" for day in range(1, 7))
    following = tuple(f"2020-01-0{day}" for day in range(2, 8))

    result = robust_calmar(
        rewards,
        decision,
        following,
        (("2020-01-01", "2020-01-03"), ("2020-01-03", "2020-01-05")),
    )
    assert len(result["folds"]) == 2
    assert result["full"]["transition_count"] == 6
