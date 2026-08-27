from __future__ import annotations

from pathlib import Path

from core.strategy_config import load_strategy_config


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "v94_dividend_quality_no_timing.json"
)


def test_v94_is_non_small_cap_dividend_and_has_no_timing():
    loaded = load_strategy_config(CONFIG)
    config = loaded["individual_config"]
    assert config["timing_enabled"] is False
    assert config["trend_risk_overlay"]["enabled"] is False
    assert config["cash_reserve_ratio"] == 0.0
    assert "PreCloseMarketCap" not in config["weights"]
    assert "TrueMarketCap" not in config["weights"]
    assert config["filter_factors"]["FilterMarketCapTop40Pct"] is True
    assert (
        config["filter_factors"]["FilterDividendYieldTop50Positive252PIT"]
        is True
    )
    assert (
        config["filter_factors"]["FilterDividendPaidAtLeast2Of3YearsPIT"]
        is True
    )


def test_v94_factor_and_filter_classes_are_registered():
    loaded = load_strategy_config(CONFIG)
    assert {factor.__name__ for factor in loaded["factor_classes"]} == {
        "LongHorizonVol36MStrict",
        "CompletedDividendConsistency3YPIT",
    }
    assert {
        factor.__name__ for factor in loaded["filter_factor_classes"]
    } >= {
        "FilterMarketCapTop40Pct",
        "FilterDividendYieldTop50Positive252PIT",
        "FilterDividendPaidAtLeast2Of3YearsPIT",
        "FilterPositiveEarningsAndCashFlow",
    }
