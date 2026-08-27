from __future__ import annotations

import numpy as np

from factor_db.factors.FinancialFilters import (
    FilterFinancialCoreCoverage,
    FilterFinancialQualityFloor,
    FilterPositiveEarnings,
    FilterPositiveEarningsAndCashFlow,
    FilterPositiveOperatingCashFlow,
    FilterPositiveROE,
)


def _panel() -> dict:
    return {
        "open": np.full((2, 4), 5.0),
        "st_mask": np.zeros((2, 4), dtype=bool),
        "eps": np.array(
            [[1.0, -1.0, 1.0, np.nan], [1.0, 1.0, 1.0, 1.0]]
        ),
        "operating_cf_ps": np.array(
            [[1.0, 1.0, -1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
        ),
        "roe": np.array(
            [[10.0, 10.0, -1.0, 10.0], [10.0, 10.0, 10.0, 10.0]]
        ),
        "gross_margin": np.array(
            [[30.0, 30.0, 30.0, np.nan], [30.0, 30.0, 30.0, 30.0]]
        ),
    }


def _kept(factor) -> list[bool]:
    return np.isfinite(factor.calc_batch(_panel())[0]).tolist()


def test_financial_filters_apply_the_declared_thresholds():
    assert _kept(FilterFinancialCoreCoverage()) == [
        True, True, True, False
    ]
    assert _kept(FilterPositiveEarnings()) == [
        True, False, True, False
    ]
    assert _kept(FilterPositiveOperatingCashFlow()) == [
        True, True, False, True
    ]
    assert _kept(FilterPositiveEarningsAndCashFlow()) == [
        True, False, False, False
    ]
    assert _kept(FilterPositiveROE()) == [
        True, True, False, True
    ]
    assert _kept(FilterFinancialQualityFloor()) == [
        True, False, False, False
    ]


def test_future_financial_rows_cannot_change_current_filter():
    factor = FilterFinancialQualityFloor()
    panel = _panel()
    expected = factor.calc_batch(panel)[0].copy()

    for field in (
        "eps",
        "operating_cf_ps",
        "roe",
        "gross_margin",
    ):
        changed = {key: value.copy() for key, value in panel.items()}
        changed[field][1] *= -100
        np.testing.assert_array_equal(
            np.isfinite(factor.calc_batch(changed)[0]),
            np.isfinite(expected),
        )
