from datetime import datetime

import numpy as np

from core.backtest import _compute_factor_scores
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


class _AlreadyRankedFactor:
    hist_days = 0
    scores_are_ranks = True

    def calc_batch(self, panel):
        return np.array([[0.9, 0.5, np.nan]], dtype=np.float32)


def test_backtest_preserves_opt_in_rank_scores_and_nan_filtering():
    result = _compute_factor_scores(
        [datetime(2020, 1, 2)],
        ["000001", "000002", "000003"],
        {"_AlreadyRankedFactor": 1.0},
        [_AlreadyRankedFactor],
        data={
            "trade_dates": np.array(["2020-01-02"], dtype="datetime64[D]"),
            "stock_codes": np.array(["000001", "000002", "000003"]),
            "open": np.ones((1, 3), dtype=np.float32),
        },
    )
    _, scores, masks, *_ = result

    np.testing.assert_array_equal(
        scores["_AlreadyRankedFactor"][0],
        np.array([0.9, 0.5, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        masks["_nan_union"][0],
        np.array([True, True, False]),
    )
    assert scores.pre_ranked_names == {"_AlreadyRankedFactor"}


def test_financial_neutral_profile_keeps_v31_execution_controls():
    from core.ga._profiles import get_profile

    profile = get_profile("v76_v31_financial_neutral_train20bp")
    assert [factor.__name__ for factor in profile["factor_classes"]] == [
        "PreCloseMarketCap",
        "AmihudIlliquidityStrict",
        "TrendReversalPreCloseStrict",
        "VolumeCVStrict",
        "FinancialQualityGrowthNeutralPIT",
    ]
    assert [
        factor.__name__ for factor in profile["filter_factor_classes"]
    ] == ["FilterST", "FilterStarST", "FilterLowPrice"]
    assert profile["fixed_parameters"]["slippage_bps"] == 20.0
    assert profile["fixed_parameters"]["retention_rank_n"] == 30
