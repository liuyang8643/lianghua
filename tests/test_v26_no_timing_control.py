import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v26_differs_from_v25_only_by_disabled_timing_settings():
    results = ROOT / "results/strategy_opt_20260721"
    v25 = json.loads(
        (results / "v25_completed_timing_control_config.json").read_text(encoding="utf-8")
    )["individual_config"]
    v26 = json.loads(
        (results / "v26_no_timing_control_config.json").read_text(encoding="utf-8")
    )["individual_config"]

    assert v26["weights"] == v25["weights"]
    for key in (
        "buy_n", "sell_m", "filter_factors", "stock_pool", "holding_period",
        "rebalance", "limit_up_protection", "timing_enabled",
        "cash_reserve_ratio", "empty_months",
    ):
        assert v26[key] == v25[key]
    assert v25["trend_risk_overlay"]["enabled"] is True
    assert v26["trend_risk_overlay"] == {
        "enabled": False,
        "mode": "dual_completed",
    }
