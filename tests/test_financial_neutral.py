import numpy as np

from factor_db.factors.FinancialNeutral import (
    FinancialQualityGrowthNeutralPIT,
)


def _panel() -> dict:
    return {
        "open": np.full((2, 3), 10.0),
        "st_mask": np.zeros((2, 3), dtype=bool),
        "eps": np.array([[1.0, 1.0, np.nan], [1.0, 1.0, np.nan]]),
        "operating_cf_ps": np.array(
            [[2.0, -1.0, np.nan], [2.0, -1.0, np.nan]]
        ),
        "gross_margin": np.array(
            [[60.0, 10.0, np.nan], [60.0, 10.0, np.nan]]
        ),
        "profit_yoy": np.array(
            [[40.0, -20.0, np.nan], [40.0, -20.0, np.nan]]
        ),
    }


def test_financial_neutral_keeps_missing_report_at_midpoint():
    score = FinancialQualityGrowthNeutralPIT().calc_batch(_panel())

    assert score[0, 0] > 0.5
    assert score[0, 1] < 0.5
    assert score[0, 2] == 0.5


def test_financial_neutral_is_future_invariant_and_preserves_base_invalid_nan():
    panel = _panel()
    original = FinancialQualityGrowthNeutralPIT().calc_batch(panel)
    panel["profit_yoy"][1] = np.array([-999.0, 999.0, 123.0])
    mutated = FinancialQualityGrowthNeutralPIT().calc_batch(panel)
    panel["open"][0, 2] = np.nan
    invalid = FinancialQualityGrowthNeutralPIT().calc_batch(panel)

    np.testing.assert_array_equal(original[0], mutated[0])
    assert np.isnan(invalid[0, 2])
