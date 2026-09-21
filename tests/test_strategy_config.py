import json
from pathlib import Path

import pytest

from ai.ga import (
    generate_initial_configs,
    get_mode_configs,
    get_profile,
    get_profile_weight_search_spaces,
    resolve_profile_name,
)
from ai.ga.config import canonicalize_individual_config
from env.action_schema import ActionSchema, SERIALIZED_DAY_CONFIG_FIELDS


ROOT = Path(__file__).resolve().parents[1]


def _raw_config() -> dict:
    return json.loads((ROOT / "configs/config.json").read_text("utf-8"))[
        "individual_config"
    ]


def test_current4_is_the_only_ga_profile_and_uses_production_vocabulary():
    assert resolve_profile_name() == "current4"
    assert "ga" in get_mode_configs()
    profile = get_profile()
    assert profile["name"] == "current4"
    assert tuple(f.__name__ for f in profile["factor_classes"]) == ActionSchema().factor_names
    assert "seed_weights" not in profile
    assert set(get_profile_weight_search_spaces()) == set(ActionSchema().factor_names)
    assert profile["search_spaces"] == {
        "turnover_rate": [0.05, 0.2],
    }
    with pytest.raises(ValueError, match="only supports"):
        get_profile("v9_dual_shadow")


def test_config_json_loads_as_one_canonical_day_config():
    raw = json.loads((ROOT / "configs/config.json").read_text("utf-8"))
    config, day = canonicalize_individual_config(raw)
    assert set(config) == set(SERIALIZED_DAY_CONFIG_FIELDS)
    assert "prefilter_n" not in config
    assert sum(config["weights"].values()) == pytest.approx(1.0)
    assert config["single_buy_pct"] == pytest.approx(1.0 / config["buy_n"])
    assert tuple(day.factor_weights) == ActionSchema().factor_names
    assert tuple(config["filter_factors"]) == ActionSchema().filter_names


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("position_multiplier", 1.0),
        ("target_exposure", 1.0),
        ("cash_reserve_ratio", 0.0),
        ("holding_period", 1),
        ("timing_enabled", False),
        ("rebalance", True),
        ("trend_risk_overlay", {"enabled": False}),
    ),
)
def test_removed_controls_fail_closed(name, value):
    config = _raw_config()
    config[name] = value
    with pytest.raises(ValueError, match="unexpected fields"):
        canonicalize_individual_config(config)[0]


def test_prefilter_is_removed_before_day_config_decoding():
    config = _raw_config()
    assert config["prefilter_n"] == 300
    canonical = canonicalize_individual_config(config)[0]
    assert "prefilter_n" not in canonical


def test_partial_or_foreign_factor_vocabulary_fails_closed():
    with pytest.raises(ValueError, match="factor vocabulary"):
        canonicalize_individual_config({"weights": {"Score": 1.0}, "buy_n": 20})


def test_ga_samples_every_current4_weight_and_roundtrips_action_schema():
    schema = ActionSchema()
    configs = generate_initial_configs(100)
    assert all(set(config) == set(SERIALIZED_DAY_CONFIG_FIELDS) for config in configs)
    assert all(0 <= weight <= 1 for config in configs for weight in config["weights"].values())
    assert any(sum(config["weights"].values()) > 1.0 for config in configs)
    assert all(any(config["factor_enabled"].values()) for config in configs)
    assert all(config["buy_n"] == 20 for config in configs)
    assert all(0.0 <= config["turnover_rate"] <= 0.2 for config in configs)
    assert len({config["turnover_rate"] for config in configs}) == len(configs)
    assert any(config["turnover_rate"] * 50 != int(config["turnover_rate"] * 50) for config in configs)
    assert all(
        config["single_buy_pct"] == 1 / config["buy_n"]
        for config in configs
    )
    assert len({tuple(config["weights"].values()) for config in configs}) > 1
    for config in configs:
        decoded = schema.from_serialized_day_config(config)
        assert schema.to_static_config(decoded) == config


def test_non_current_profile_in_wrapper_fails_closed(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps({"ga_profile": "core6", "individual_config": _raw_config()}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only supports"):
        canonicalize_individual_config(json.loads(path.read_text("utf-8")))
