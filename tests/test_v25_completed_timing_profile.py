import json
from pathlib import Path

from core.ga._profiles import get_profile


ROOT = Path(__file__).resolve().parents[1]


def test_v25_profile_is_strict_completed_and_contains_no_gap_factor():
    profile = get_profile("v25_preclose_official_strict_completed_timing")
    names = [factor.__name__ for factor in profile["factor_classes"]]
    settings = profile["fixed_parameters"]["trend_risk_overlay"]

    assert names == [
        "PreCloseMarketCap",
        "AmihudIlliquidityStrict",
        "TrendReversalPreCloseStrict",
        "VolumeCVStrict",
    ]
    assert all("Gap" not in name for name in names)
    assert settings["mode"] == "dual_completed"
    assert settings["strict_history"] is True
    assert settings["strict_warmup_multiplier"] == 1.0
    assert profile["fixed_parameters"]["stock_pool"] == ["60", "00", "30"]


def test_v25_frozen_config_does_not_restore_gap_derived_timing():
    path = ROOT / "results/strategy_opt_20260721/v25_completed_timing_control_config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    individual = payload["individual_config"]

    assert individual["trend_risk_overlay"]["mode"] == "dual_completed"
    assert all("Gap" not in name for name in individual["weights"])
    assert "TrueMarketCap" not in individual["weights"]
