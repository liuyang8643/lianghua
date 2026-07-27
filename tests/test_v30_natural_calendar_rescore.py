import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from research_train_robustness import anchored_metrics
from research_v30_natural_calendar_rescore import (
    EXPECTED_CANDIDATE_COUNT,
    FOLD_LABELS,
    _load_closed_candidates,
    _result_row,
    _compute_research_arrays,
    _validate_independent_canary,
    natural_calendar_metrics,
    strict_pareto,
)


def _training_dates():
    start = date(2010, 1, 4)
    stop = date(2018, 12, 28)
    values = []
    current = start
    while current <= stop:
        if current.weekday() < 5:
            values.append(current.isoformat())
        current += timedelta(days=1)
    # The production calendar has holiday gaps.  Use its exact dates from a
    # previously audited training record so date validation stays realistic.
    record = json.loads(
        Path("results/strategy_opt_20260721/v25_completed_timing_train/record.json")
        .read_text(encoding="utf-8")
    )
    return record["dates"]


def test_natural_calendar_metrics_uses_exact_anchored_folds():
    dates = _training_dates()
    daily = np.linspace(-0.7, 0.9, len(dates))
    result = natural_calendar_metrics(daily, dates)
    assert tuple(result["folds"]) == FOLD_LABELS
    assert result["full"] == anchored_metrics(daily)
    for start, label in zip((2010, 2013, 2016), FOLD_LABELS):
        mask = np.array([start <= int(value[:4]) <= start + 2 for value in dates])
        assert result["folds"][label] == anchored_metrics(daily[mask])
    assert result["robust_score"] == pytest.approx(
        0.5 * result["full"]["calmar"]
        + 0.5 * min(result["fold_calmars"])
    )


def test_natural_calendar_metrics_rejects_holdout_or_misalignment():
    dates = _training_dates()
    with pytest.raises(ValueError, match="daily returns"):
        natural_calendar_metrics(np.zeros(len(dates) - 1), dates)
    contaminated = list(dates)
    contaminated[-1] = "2019-01-02"
    with pytest.raises(ValueError, match="training dates|holdout"):
        natural_calendar_metrics(np.zeros(len(contaminated)), contaminated)


def test_strict_pareto_has_no_tolerance():
    rows = [
        {"id": "a", "x": 1.0, "y": 1.0},
        {"id": "b", "x": 1.0, "y": 1.0 + 1e-15},
        {"id": "c", "x": 1.1, "y": 0.9},
    ]
    assert [row["id"] for row in strict_pareto(rows, ("x", "y"))] == ["b", "c"]
    with pytest.raises(ValueError, match="non-finite"):
        strict_pareto([{"x": np.nan, "y": 1.0}], ("x", "y"))


def test_closed_v28_candidate_identity_and_count():
    rows = _load_closed_candidates(
        Path("results/strategy_opt_20260721/v28_weight_candidates.json")
    )
    assert len(rows) == EXPECTED_CANDIDATE_COUNT
    assert len({json.dumps(row["individual_config"], sort_keys=True) for row in rows}) == len(rows)


def test_result_row_requires_exact_robust_formula():
    result = {
        "fold_calmars": [1.0, 2.0, 3.0],
        "calmar": 2.0,
        "raw_fitness": 1.5,
        "fitness": 1.5,
        "average_exposure": 0.6,
        "exposure_constraint_passed": True,
        "annualized": 20.0,
        "max_drawdown": -10.0,
        "sharpe": 1.0,
        "total_return": 100.0,
    }
    row = _result_row(result, {"candidate_index": 0, "label": "x", "individual_config": {}})
    assert row["worst_fold_calmar"] == 1.0
    bad = dict(result, raw_fitness=1.5000001)
    with pytest.raises(ValueError, match="robust score"):
        _result_row(bad, {"candidate_index": 0})


def test_independent_canary_is_fail_closed():
    rows = [
        {
            "full_calmar": 0.0,
            "fold_calmars": {label: 0.0 for label in FOLD_LABELS},
        }
        for _ in range(EXPECTED_CANDIDATE_COUNT)
    ]
    rows[189] = {
        "full_calmar": 2.9743907885680705,
        "fold_calmars": {
            "2010-2012": 1.5856135393323012,
            "2013-2015": 8.884354962146913,
            "2016-2018": 0.8744083516850107,
        },
    }
    _validate_independent_canary(rows)
    rows[189]["fold_calmars"]["2016-2018"] += 1e-9
    with pytest.raises(ValueError, match="canary mismatch"):
        _validate_independent_canary(rows)


def test_factor_ranks_are_denominated_only_by_strategy_pool():
    class FixedFactor:
        hist_days = 1

        def calc_batch(self, data):
            return np.asarray([[10.0, 20.0, 1000.0]], dtype=float)

    data = {
        "stock_codes": np.asarray(["000001.SZ", "600000.SH", "688001.SH"]),
        "trade_dates": np.asarray(["2018-12-28"], dtype="datetime64[D]"),
    }
    arrays, keys = _compute_research_arrays(
        data, [FixedFactor], [], ("60", "00", "30")
    )
    assert keys == {"FixedFactor"}
    expected = np.zeros((1, 3), dtype=np.float32)
    from core.scoring import scores_to_ranks

    expected[:, :2] = scores_to_ranks(np.asarray([[10.0, 20.0]], dtype=np.float32))
    np.testing.assert_array_equal(arrays["FixedFactor"], expected)
    assert arrays["FixedFactor"][0, 2] == 0.0
