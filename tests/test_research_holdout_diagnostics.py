import pytest

from research_holdout_diagnostics import PERIODS, _validate_period_dates


@pytest.mark.parametrize("scope", PERIODS)
def test_holdout_period_guard_accepts_only_frozen_endpoints(scope):
    first, last = PERIODS[scope]

    _validate_period_dates([first, last], scope)

    with pytest.raises(ValueError, match="exact period"):
        _validate_period_dates([first, "2099-12-31"], scope)


def test_holdout_period_guard_rejects_duplicates_and_unknown_scope():
    first, _ = PERIODS["validation"]
    with pytest.raises(ValueError, match="strictly increasing"):
        _validate_period_dates([first, first], "validation")
    with pytest.raises(ValueError, match="unsupported"):
        _validate_period_dates([first], "future")
