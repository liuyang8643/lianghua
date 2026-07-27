import numpy as np
import pytest

from research_train_robustness import (
    EXPECTED_FIRST_DATE,
    EXPECTED_LAST_DATE,
    _drawdown_episode,
    _rolling_blocks,
    _validate_train_dates,
    anchored_metrics,
)


def test_first_day_loss_is_included_in_drawdown():
    metrics = anchored_metrics([-10.0, 0.0])

    assert metrics["max_drawdown"] == pytest.approx(-10.0)
    assert metrics["terminal_nav"] == pytest.approx(0.9)


def test_metrics_reject_nonfinite_and_total_loss():
    with pytest.raises(ValueError, match="non-finite"):
        anchored_metrics([0.0, np.nan])
    with pytest.raises(ValueError, match="-100%"):
        anchored_metrics([-100.0, 0.0])


def test_training_date_guard_rejects_holdout_or_partial_period():
    _validate_train_dates([EXPECTED_FIRST_DATE, EXPECTED_LAST_DATE])

    with pytest.raises(ValueError, match="fixed 2010-2018"):
        _validate_train_dates([EXPECTED_FIRST_DATE, "2019-01-02"])
    with pytest.raises(ValueError, match="strictly increasing"):
        _validate_train_dates([EXPECTED_FIRST_DATE, EXPECTED_FIRST_DATE])


def test_rolling_blocks_include_the_final_possible_window(monkeypatch):
    monkeypatch.setattr("research_train_robustness.ROLLING_WINDOW_DAYS", 4)
    monkeypatch.setattr("research_train_robustness.ROLLING_STEP_DAYS", 3)
    dates = [f"2010-01-{day:02d}" for day in range(1, 9)]

    result = _rolling_blocks(dates, np.full(8, 0.1))

    assert result["count"] == 3
    assert result["windows"][-1]["start"] == dates[4]
    assert result["windows"][-1]["end"] == dates[7]


def test_drawdown_episode_can_start_at_initial_nav():
    dates = ["2010-01-04", "2010-01-05", "2010-01-06"]

    episode = _drawdown_episode(dates, np.array([-10.0, 5.0, 6.0]))

    assert episode["peak_date"] == "initial_nav"
    assert episode["trough_date"] == "2010-01-04"
    assert episode["drawdown_pct"] == pytest.approx(-10.0)
