import numpy as np
import pytest

from factor_db.factors.AmountCeilingFilter import (
    FilterMeanAmount20Max40M,
    FilterMeanAmount20Max45M,
    FilterMeanAmount20Max50M,
    FilterMeanAmount20Max55M,
    FilterMeanAmount20Max60M,
    FilterMeanAmount20Max70M,
    FilterMeanAmount20Max80M,
    FilterMeanAmount20Max100M,
)


@pytest.mark.parametrize(
    ("factor_class", "limit"),
    [
        (FilterMeanAmount20Max40M, 40_000_000.0),
        (FilterMeanAmount20Max45M, 45_000_000.0),
        (FilterMeanAmount20Max50M, 50_000_000.0),
        (FilterMeanAmount20Max55M, 55_000_000.0),
        (FilterMeanAmount20Max60M, 60_000_000.0),
        (FilterMeanAmount20Max70M, 70_000_000.0),
        (FilterMeanAmount20Max80M, 80_000_000.0),
        (FilterMeanAmount20Max100M, 100_000_000.0),
    ],
)
def test_amount_ceiling_uses_exactly_20_completed_days(factor_class, limit):
    amount = np.full((22, 3), limit)
    amount[:20, 1] = limit + 1.0
    amount[:20, 2] = limit - 1.0

    actual = factor_class().calc_batch({"amount": amount})

    assert np.all(np.isnan(actual[:20]))
    np.testing.assert_array_equal(
        np.isfinite(actual[20]), [True, False, True],
    )


def test_amount_ceiling_does_not_use_current_day_amount():
    amount = np.full((22, 1), 10_000_000.0)
    expected = FilterMeanAmount20Max80M().calc_batch({"amount": amount})
    amount[20, 0] = 2_000_000_000.0

    actual = FilterMeanAmount20Max80M().calc_batch({"amount": amount})

    assert actual[20, 0] == expected[20, 0]
    assert np.isnan(actual[21, 0])


def test_amount_ceiling_exposes_incomplete_windows():
    amount = np.full((21, 1), 10_000_000.0)
    amount[7, 0] = np.nan

    actual = FilterMeanAmount20Max80M().calc_batch({"amount": amount})

    assert np.isnan(actual[20, 0])
