from core.ga._profiles import get_profile


def test_v71_profile_keeps_v31_execution_and_searches_only_financial_weights():
    profile = get_profile("v71_v31_financial_pit_train20bp")
    names = [factor.__name__ for factor in profile["factor_classes"]]

    assert names[:4] == [
        "PreCloseMarketCap",
        "AmihudIlliquidityStrict",
        "TrendReversalPreCloseStrict",
        "VolumeCVStrict",
    ]
    assert names[4:] == [
        "GrossMarginQuality",
        "CashFlowCoverage",
        "ROEQuality",
        "ProfitGrowth",
        "EarningsQualityComposite",
    ]
    assert [
        factor.__name__
        for factor in profile["filter_factor_classes"]
    ] == [
        "FilterST",
        "FilterStarST",
        "FilterLowPrice",
        "FilterFinancialCoreCoverage",
        "FilterPositiveEarnings",
        "FilterPositiveOperatingCashFlow",
        "FilterPositiveEarningsAndCashFlow",
        "FilterPositiveROE",
        "FilterFinancialQualityFloor",
    ]
    assert profile["fixed_weights"] == {
        "PreCloseMarketCap": 1.0,
        "AmihudIlliquidityStrict": 0.4,
        "TrendReversalPreCloseStrict": 0.1,
        "VolumeCVStrict": 0.2,
    }
    assert profile["fixed_parameters"]["slippage_bps"] == 20.0
    assert profile["fixed_parameters"]["retention_rank_n"] == 30
    assert profile["search_spaces"] == {"buy_n": [30]}
