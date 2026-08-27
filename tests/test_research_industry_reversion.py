from __future__ import annotations

import numpy as np

from testback.research_industry_reversion import (
    IndustryCandidate,
    candidate_grid,
    derive_periods,
    robust_training_fitness,
    select_training_candidate,
)


def test_default_grid_is_small_deterministic_and_cost_free():
    first = candidate_grid()
    second = candidate_grid()
    assert first == second
    assert len(first) == 24
    assert {candidate.level for candidate in first} == {"l2", "l3"}
    assert {candidate.buy_n for candidate in first} == {10, 20}
    assert {candidate.holding_period for candidate in first} == {5, 10}
    assert all(candidate.sell_m == candidate.buy_n * 2 for candidate in first)
    assert all("cost" not in candidate.__dict__ for candidate in first)


def test_standard_split_matches_repository_protocol_when_coverage_allows():
    dates = np.arange(
        np.datetime64("1990-01-01"),
        np.datetime64("2026-08-26"),
        dtype="datetime64[D]",
    )
    periods = derive_periods(dates, np.datetime64("1990-01-01"))
    assert periods["train"].start == np.datetime64("2010-01-01")
    assert periods["train"].end == np.datetime64("2018-12-31")
    assert periods["validation"].start == np.datetime64("2019-01-01")
    assert periods["validation"].end == np.datetime64("2022-12-31")
    assert periods["test"].start == np.datetime64("2023-01-01")
    assert periods["test"].end == np.datetime64("2026-08-25")


def test_short_coverage_falls_back_to_contiguous_availability_split():
    dates = np.arange(
        np.datetime64("2020-01-01"),
        np.datetime64("2020-04-10"),
        dtype="datetime64[D]",
    )
    periods = derive_periods(dates, np.datetime64("2019-01-01"))
    assert periods["train"].end < periods["validation"].start
    assert periods["validation"].end < periods["test"].start
    assert periods["train"].start == dates[0]
    assert periods["test"].end == dates[-1]


def test_robust_fitness_penalizes_one_bad_chronological_fold():
    stable = np.tile([0.1, -0.02], 60)
    unstable = stable.copy()
    unstable[-40:] = -0.2
    _, stable_fitness, stable_folds = robust_training_fitness(stable)
    _, unstable_fitness, unstable_folds = robust_training_fitness(unstable)
    assert len(stable_folds) == 3
    assert len(unstable_folds) == 3
    assert stable_fitness > unstable_fitness


def _record(candidate: IndustryCandidate, fitness: float, exposure: float) -> dict:
    return {
        "candidate": candidate.__dict__.copy(),
        "fitness": fitness,
        "fold_calmars": [fitness, fitness],
        "calmar": fitness,
        "sharpe": fitness,
        "average_exposure": exposure,
    }


def test_training_selection_rejects_cash_like_high_metric_candidate():
    candidates = candidate_grid()
    cash_like = _record(candidates[0], fitness=100.0, exposure=0.01)
    invested = _record(candidates[1], fitness=1.0, exposure=0.90)
    selected = select_training_candidate([cash_like, invested])
    assert selected["candidate"] == invested["candidate"]


def test_training_selection_uses_only_fields_in_training_records():
    candidates = candidate_grid()
    better = _record(candidates[0], fitness=2.0, exposure=0.90)
    worse = _record(candidates[1], fitness=1.0, exposure=0.90)
    better["test_calmar"] = -1e9
    worse["test_calmar"] = 1e9
    assert select_training_candidate([better, worse])["candidate"] == better["candidate"]
