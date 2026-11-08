import json
from pathlib import Path

import pytest

from ai.ga.config import canonicalize_individual_config
from env.fees import DEFAULT_FEE_SCHEDULE


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return json.loads((ROOT / "configs/config.json").read_text("utf-8"))[
        "individual_config"
    ]


def test_slippage_is_fixed_environment_semantics():
    config = canonicalize_individual_config(_config())[0]
    assert "slippage_bps" not in config
    assert DEFAULT_FEE_SCHEDULE.slippage_rate == pytest.approx(0.0025)


@pytest.mark.parametrize("value", [10.0, 20.0, 50.0])
def test_day_config_rejects_slippage_override(value):
    config = _config()
    config["slippage_bps"] = value
    with pytest.raises(ValueError, match="unexpected fields: slippage_bps"):
        canonicalize_individual_config(config)


@pytest.mark.parametrize("value", [True, -0.01, 1.0, float("nan"), float("inf")])
def test_invalid_rebalance_band_fails_closed(value):
    config = _config()
    config["rebalance_band_pct"] = value
    with pytest.raises((TypeError, ValueError), match="rebalance_band_pct"):
        canonicalize_individual_config(config)
