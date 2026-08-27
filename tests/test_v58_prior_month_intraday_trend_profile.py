import json
from pathlib import Path

from core.factors.registry import get_factor_class
from core.ga import build_individual_config, get_profile
from core.strategy_config import normalize_individual_config
from factor_db.factors.CompletedPriorMonthIntradayTrendStrict import (
    CompletedPriorMonthIntradayTrendStrict,
)
from testback.run_ga import _load_candidate_configs


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = (
    ROOT
    / "configs"
    / "v58_prior_month_intraday_trend_sleeves_train.json"
)

V31_WEIGHTS = {
    "PreCloseMarketCap": 1.0,
    "AmihudIlliquidityStrict": 0.4,
    "TrendReversalPreCloseStrict": 0.1,
    "VolumeCVStrict": 0.2,
}
ALL_WEIGHTS = {
    **V31_WEIGHTS,
    "CompletedPriorMonthIntradayTrendStrict": 0.0,
}
V31_FILTERS = {
    "FilterST": True,
    "FilterStarST": True,
    "FilterLowPrice": True,
}
V31_OVERLAY = {
    "enabled": True,
    "mode": "dual_completed",
    "strict_history": True,
    "strict_warmup_multiplier": 1.0,
    "floor": 0.0,
    "ceiling": 1.0,
    "momentum_center": -0.055,
    "momentum_scale": 0.012,
    "ma_center": 1.01,
    "ma_scale": 0.009,
    "softmin_sharpness": 4.0,
    "slope": 2.0,
    "momentum_window": 10,
    "ma_window": 20,
    "strategy_weight": 0.8,
    "strategy_momentum_window": 5,
    "strategy_momentum_center": -0.044,
    "strategy_momentum_scale": 0.015,
    "strategy_ma_window": 20,
    "strategy_ma_center": 1.014,
    "strategy_ma_scale": 0.009,
    "strategy_softmin_sharpness": 4.0,
    "strategy_slope": 2.0,
}


def test_v58_profile_fixes_v31_semantics_costs_and_training_period():
    profile = get_profile("v58_prior_month_intraday_trend_sleeves")

    assert [factor.__name__ for factor in profile["factor_classes"]] == [
        "PreCloseMarketCap",
        "AmihudIlliquidityStrict",
        "TrendReversalPreCloseStrict",
        "VolumeCVStrict",
        "CompletedPriorMonthIntradayTrendStrict",
    ]
    assert [
        factor.__name__ for factor in profile["filter_factor_classes"]
    ] == ["FilterST", "FilterStarST", "FilterLowPrice"]
    assert profile["weight_search_spaces"] is None
    assert profile["fixed_weights"] == ALL_WEIGHTS
    assert profile["search_spaces"] == {}
    fixed = profile["fixed_parameters"]
    assert fixed["buy_n"] == 30
    assert fixed["stock_pool"] == ["60", "00", "30"]
    assert fixed["holding_period"] == 1
    assert fixed["retention_rank_n"] == 30
    assert fixed["retention_mode"] == "expanded_equal_weight"
    assert fixed["rebalance_band_pct"] == 0.15
    assert fixed["slippage_bps"] == 20.0
    assert fixed["rebalance"] is True
    assert fixed["limit_up_protection"] is True
    assert fixed["cash_reserve_ratio"] == 0.0
    assert fixed["timing_enabled"] is False
    assert fixed["filter_factors"] == V31_FILTERS
    assert fixed["trend_risk_overlay"] == V31_OVERLAY
    assert "selection_sleeves" not in fixed
    assert profile["constraints"] == {"sell_m_equals_buy_n": True}
    assert profile["training_objective"] == {
        "mode": "robust_calmar",
        "full_weight": 0.5,
        "min_average_exposure": 0.45,
        "calendar_folds": [
            ["2010-01-01", "2012-12-31"],
            ["2013-01-01", "2015-12-31"],
            ["2016-01-01", "2018-12-31"],
        ],
    }
    assert str(profile["preload_start_date"]) == "2010-01-01"
    assert str(profile["preload_end_date"]) == "2018-12-31"

    config = build_individual_config(
        profile_name="v58_prior_month_intraday_trend_sleeves"
    )
    assert config["buy_n"] == 30
    assert config["sell_m"] == 30
    assert config["slippage_bps"] == 20.0
    assert config["rebalance_band_pct"] == 0.15
    assert config["retention_rank_n"] == 30
    assert config["retention_mode"] == "expanded_equal_weight"
    assert "selection_sleeves" not in config


def test_v58_factor_is_auto_discovered_without_manual_registry_state():
    assert (
        get_factor_class("CompletedPriorMonthIntradayTrendStrict")
        is CompletedPriorMonthIntradayTrendStrict
    )


def test_v58_training_candidates_are_four_complete_fixed_semantic_arms():
    payload = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    assert payload["ga_profile"] == (
        "v58_prior_month_intraday_trend_sleeves"
    )
    assert payload["selection_scope"] == "training_only_2010_2018"
    assert payload["inherit_profile_defaults"] is False
    assert [item["name"] for item in payload["configs"]] == [
        "control_core30_trend0",
        "core29_trend1",
        "core28_trend2",
        "core27_trend3",
    ]

    configs = _load_candidate_configs(
        CANDIDATES,
        profile_name="v58_prior_month_intraday_trend_sleeves",
    )
    assert len(configs) == 4
    reference_without_sleeves = {
        key: value
        for key, value in configs[0].items()
        if key != "selection_sleeves"
    }

    required_keys = {
        "weights",
        "buy_n",
        "sell_m",
        "filter_factors",
        "stock_pool",
        "holding_period",
        "retention_rank_n",
        "retention_mode",
        "rebalance_band_pct",
        "slippage_bps",
        "rebalance",
        "limit_up_protection",
        "selection_sleeves",
        "trend_risk_overlay",
        "timing_enabled",
        "cash_reserve_ratio",
        "empty_months",
    }
    for config in configs:
        assert required_keys <= config.keys()
        assert normalize_individual_config(
            config,
            "v58_prior_month_intraday_trend_sleeves",
        ) == config
        assert config["weights"] == ALL_WEIGHTS
        assert config["buy_n"] == config["sell_m"] == 30
        assert config["filter_factors"] == V31_FILTERS
        assert config["stock_pool"] == ["60", "00", "30"]
        assert config["holding_period"] == 1
        assert config["retention_rank_n"] == 30
        assert config["retention_mode"] == "expanded_equal_weight"
        assert config["rebalance_band_pct"] == 0.15
        assert config["slippage_bps"] == 20.0
        assert config["rebalance"] is True
        assert config["limit_up_protection"] is True
        assert config["trend_risk_overlay"] == V31_OVERLAY
        assert config["timing_enabled"] is False
        assert config["cash_reserve_ratio"] == 0.0
        assert config["empty_months"] is None
        assert {
            key: value
            for key, value in config.items()
            if key != "selection_sleeves"
        } == reference_without_sleeves

    for trend_slots, config in enumerate(configs):
        sleeves = config["selection_sleeves"]
        core = sleeves[0]
        assert core == {
            "name": "v31_core",
            "slots": 30 - trend_slots,
            "weights": V31_WEIGHTS,
        }
        assert sum(sleeve["slots"] for sleeve in sleeves) == 30
        if trend_slots == 0:
            assert sleeves == [core]
        else:
            assert sleeves[1] == {
                "name": "positive_prior_month_intraday_trend",
                "slots": trend_slots,
                "weights": {
                    "CompletedPriorMonthIntradayTrendStrict": 1.0
                },
            }


def test_v58_candidate_payload_contains_no_holdout_configuration():
    payload = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    forbidden_keys = {
        "validation",
        "validation_start",
        "validation_end",
        "test",
        "test_start",
        "test_end",
        "holdout",
        "sealed_holdout",
        "split_period_results",
    }

    def visit(value):
        if isinstance(value, dict):
            assert forbidden_keys.isdisjoint(value)
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(payload)
