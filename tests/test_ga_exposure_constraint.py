import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from env.backtest import EpisodeSession, prepare_episode_from_runtime, run_day_config_episode
from env.action_schema import ActionSchema
from ai.ga.config import canonicalize_ga_genes
from ai.ga.train import (
    _evaluate_individual,
    _load_candidate_configs,
    _run_ga,
)


ROOT = Path(__file__).resolve().parents[1]


from test_backtest_lightweight import write_canonical_runtime as _runtime


def _current_config() -> dict:
    raw = json.loads((ROOT / "configs" / "config.json").read_text("utf-8"))[
        "individual_config"
    ]
    schema = ActionSchema()
    return {**schema.to_static_config(schema.from_static_config(raw)), "prefilter_n": 300}


def test_candidate_loader_always_normalizes_and_action_schema_validates(tmp_path):
    path = tmp_path / "candidate.json"
    path.write_text(
        json.dumps(
            {
                "configs": [_current_config()],
            }
        ),
        encoding="utf-8",
    )

    config = _load_candidate_configs(path, profile_name="current4")[0]

    assert sum(config["weights"].values()) == pytest.approx(1.0)
    assert config["factor_enabled"] == {name: value != 0 for name, value in config["weights"].items()}
    assert config["single_buy_pct"] == pytest.approx(1.0 / config["buy_n"])
    assert "prefilter_n" not in config


@pytest.mark.parametrize(
    ("name", "value", "message"),
    (
        ("position_multipliers", [1.0], "fields must match exactly"),
        ("stock_pool", ["60"], "fields must match exactly"),
        ("selection_sleeves", [], "fields must match exactly"),
    ),
)
def test_candidate_loader_cannot_bypass_fail_closed_normalization(
    tmp_path,
    name,
    value,
    message,
):
    config = _current_config()
    config[name] = value
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "configs": [config],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        _load_candidate_configs(path, profile_name="current4")


@pytest.mark.parametrize('payload', [[], {'individual_config': {}}, {'configs': [{'individual_config': {}}]}])
def test_candidate_loader_rejects_alternative_wrappers(tmp_path, payload):
    path = tmp_path / 'candidate.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError):
        _load_candidate_configs(path, profile_name='current4')


def test_ga_worker_uses_the_same_canonical_result_and_requires_full_investment(
    tmp_path,
):
    runtime_path = tmp_path / "runtime.npz"
    _runtime(runtime_path)
    episode = prepare_episode_from_runtime(
        runtime_path,
        "2020-06-20",
        "2020-06-24",
        prefilter_n=300,
    )
    config = _current_config()
    canonical, day = canonicalize_ga_genes(config)
    fixed = run_day_config_episode(EpisodeSession(episode), lambda _: day)
    evaluated = _evaluate_individual(episode, config)

    assert evaluated["individual_config"] == canonical
    assert evaluated["total_return"] == float((fixed.nav[-1] / fixed.nav[0] - 1) * 100)
    assert evaluated["average_exposure"] == pytest.approx(
        np.mean(fixed.exposure)
    )
    assert evaluated["full_investment_contract_satisfied"] is True
    assert "exposure_constraint_passed" not in evaluated


def test_ga_contract_is_hard_failure_not_a_fitness_penalty(tmp_path, monkeypatch):
    import ai.ga.train as ga_train
    runtime_path = tmp_path / "runtime.npz"
    _runtime(runtime_path)
    episode = prepare_episode_from_runtime(runtime_path, "2020-06-20", "2020-06-24", prefilter_n=300)
    monkeypatch.setattr(ga_train, "run_day_config_episode", lambda *a, **kw:
                        SimpleNamespace(full_investment_contract_satisfied=False))
    with pytest.raises(RuntimeError, match="full-investment"):
        _evaluate_individual(episode, _current_config())


def test_cached_result_cannot_bypass_generation_full_investment_contract(tmp_path, monkeypatch):
    import ai.ga.train as ga_train
    runtime_path = tmp_path / "runtime.npz"
    _runtime(runtime_path)
    config, _ = canonicalize_ga_genes(_current_config())
    cached = {ga_train._ga_cache_key(config): {
        "individual_config": config, "calmar": 1.0,
        "full_investment_contract_satisfied": False,
    }}
    monkeypatch.setattr(ga_train, "_load_canonical_ga_resume", lambda *args: (cached, -1))
    monkeypatch.setattr(ga_train, "ga_optimizer", lambda *args, **kwargs: [config])
    monkeypatch.setattr(ga_train, "_eval_parallel", lambda *args, **kwargs:
                        pytest.fail("cached configuration was unexpectedly reevaluated"))
    output = tmp_path / "resume"
    args = SimpleNamespace(
        mode="debug", seed=17, runtime_path=runtime_path, config=ROOT / "configs/config.json",
        lookback=64, workers=1, continue_from=None,
        warm_start=None, candidate_configs=None,
    )
    dates = [datetime.fromisoformat(f"2020-06-{day:02d}") for day in range(20, 25)]
    with pytest.raises(RuntimeError, match="generation contains a non-full-investment"):
        _run_ga(args, {"population_size": 2, "generations": 1}, dates, resume_dir=output)
    assert not (output / "best_individual_config.json").exists()
    assert not (output / "all_results.jsonl").exists()


def test_current4_ga_entry_evaluates_multiple_candidates_in_spawn_workers(
    tmp_path,
):
    runtime_path = tmp_path / "runtime.npz"
    _runtime(runtime_path)
    first = _current_config()
    second = _current_config()
    second["weights"] = dict.fromkeys(ActionSchema().factor_names, 0.5)
    second["factor_enabled"] = dict.fromkeys(ActionSchema().factor_names, True)
    candidate_path = tmp_path / "candidates.json"
    candidate_path.write_text(
        json.dumps({"configs": [first, second]}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "ga"
    args = SimpleNamespace(
        mode="debug",
        output_dir=str(output_dir),
        runtime_path=str(runtime_path),
        config=str(ROOT / "configs/config.json"),
        lookback=64,
        workers=2,
        seed=17,
        candidate_configs=str(candidate_path),
        warm_start=None,
        continue_from=None,
    )
    dates = [
        datetime.fromisoformat(f"2020-06-{day:02d}")
        for day in range(20, 25)
    ]

    winner = _run_ga(
        args,
        {"population_size": 2, "generations": 1},
        dates,
        profile_name="current4",
    )

    assert winner["full_investment_contract_satisfied"] is True
    assert (output_dir / "best_individual_config.json").is_file()
    rows = [
        json.loads(line)
        for line in (output_dir / "all_results.jsonl")
        .read_text("utf-8")
        .splitlines()
    ]
    assert len(rows) == 2
    assert all(row["full_investment_contract_satisfied"] is True for row in rows)

    episode = prepare_episode_from_runtime(
        runtime_path,
        "2020-06-20",
        "2020-06-24",
        prefilter_n=300,
    )
    for row in rows:
        _, day = canonicalize_ga_genes(row["config"])
        serial = run_day_config_episode(EpisodeSession(episode), lambda _: day)
        assert row["total_return"] == float((serial.nav[-1] / serial.nav[0] - 1) * 100)
        assert row["average_exposure"] == pytest.approx(
            np.mean(serial.exposure)
        )
