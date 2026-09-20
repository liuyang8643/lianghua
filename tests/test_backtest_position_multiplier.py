import json
from pathlib import Path

import pytest

from ai.ga.config import canonicalize_individual_config
from testback.backtest import run_static_config_backtest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("position_multipliers", [1.0]),
        ("position_multiplier", 1.0),
        ("enforce_position_multiplier_on_sell_m", False),
        ("holding_period", 1),
        ("rebalance", True),
        ("rebalance_now", True),
        ("target_exposure", 1.0),
        ("target_cash", 0.0),
        ("cash_reserve_ratio", 0.0),
        ("timing_enabled", False),
        ("stock_pool", ["60", "00", "30", "688"]),
        ("selection_sleeves", []),
        ("retention_rank_n", 25),
        ("slippage_bps", 10.0),
    ),
)
def test_fixed_policy_rejects_removed_or_non_day_config_fields(name, value):
    payload = json.loads((ROOT / "configs" / "config.json").read_text("utf-8"))
    payload["individual_config"][name] = value

    with pytest.raises(ValueError):
        canonicalize_individual_config(payload)


def test_backtest_signature_has_no_hidden_position_control_entrypoint():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        run_static_config_backtest(
            None,
            {},
            position_multipliers=[1.0],
        )
