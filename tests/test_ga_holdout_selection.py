import json
from pathlib import Path

import pytest

from ai.ga.train import (
    _append_jsonl_rows,
    _ga_cache_key,
    _rebuild_from_jsonl,
    _save_training_candidate,
    _select_training_candidate,
)


ROOT = Path(__file__).resolve().parents[1]


def _current_config() -> dict:
    return json.loads((ROOT / "configs/config.json").read_text("utf-8"))[
        "individual_config"
    ]


def test_candidate_selection_uses_training_calmar_only():
    selected = _select_training_candidate(
        {
            "holdout": {
                "calmar": 1.0,
                "val_calmar": 999.0,
                "individual_config": {"name": "holdout"},
            },
            "train": {
                "calmar": 2.0,
                "val_calmar": -999.0,
                "individual_config": {"name": "train"},
            },
        }
    )
    assert selected["individual_config"]["name"] == "train"


def test_calmar_tie_is_deterministic_and_holdout_blind():
    candidates = [
        {
            "calmar": 2.0,
            "val_calmar": sign * 99.0,
            "individual_config": {"name": name},
        }
        for name, sign in (("a", 1), ("b", -1))
    ]
    expected = max(_ga_cache_key(item["individual_config"]) for item in candidates)
    forward = _select_training_candidate(dict(enumerate(candidates)))
    reverse = _select_training_candidate(dict(enumerate(reversed(candidates))))
    assert _ga_cache_key(forward["individual_config"]) == expected
    assert _ga_cache_key(reverse["individual_config"]) == expected


def test_global_training_winner_is_saved_without_holdout_fields(tmp_path):
    first = _current_config()
    second = _current_config()
    second["weights"] = {
        "AmihudIlliquidity": 0.7,
        "TrueMarketCap": 0.4,
        "VolumeCV": 0.6,
        "AmountBasedSmallCap": 0.3,
    }
    winner = {"calmar": 2.0, "individual_config": first}
    selected = _save_training_candidate(
        tmp_path,
        "current4",
        {"winner": winner, "last": {"calmar": 1.8, "individual_config": second}},
        prefilter_n=300,
    )
    saved = json.loads((tmp_path / "best_individual_config.json").read_text("utf-8"))
    assert selected is winner
    assert saved["ga_profile"] == "current4"
    assert saved["individual_config"]["prefilter_n"] == 300


@pytest.mark.parametrize(
    "extra",
    ({"val_calmar": 2.0}, {"test_calmar": 2.0}, {"fold_calmars": [1.0]}),
)
def test_resume_rejects_holdout_and_removed_objective_schemas(tmp_path, extra):
    row = {
        "generation": 0,
        "calmar": 1.0,
        "config": _current_config(),
        **extra,
    }
    _append_jsonl_rows(tmp_path / "all_results.jsonl", [row])
    with pytest.raises(ValueError):
        _rebuild_from_jsonl(tmp_path)


def test_resume_cache_uses_the_live_canonical_key(tmp_path):
    config = _current_config()
    config.pop("prefilter_n")
    equivalent = {key: config[key] for key in reversed(config)}
    equivalent["weights"] = dict(reversed(list(config["weights"].items())))
    _append_jsonl_rows(
        tmp_path / "all_results.jsonl",
        [{"generation": 4, "calmar": 1.4, "config": config, "metrics": {"calmar": 1.4},
          "total_return": 20.0, "sharpe": 1.0, "annualized": 14.0, "max_drawdown": -10.0,
          "average_exposure": .99, "full_investment_contract_satisfied": True,
          "evaluation_elapsed_seconds": 2.3, "execution_diagnostics": {"total_fees": 123.0}}],
    )
    cache, generation = _rebuild_from_jsonl(tmp_path)
    assert list(cache) == [_ga_cache_key(equivalent)]
    assert generation == 4
    assert cache[_ga_cache_key(equivalent)]['execution_diagnostics'] == {'total_fees': 123.0}
    assert cache[_ga_cache_key(equivalent)]['evaluation_elapsed_seconds'] == 2.3
