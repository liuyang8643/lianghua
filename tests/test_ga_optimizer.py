"""Current4 GA selection, diversity and reproducibility tests."""

from ai.ga import build_individual_config, generate_initial_configs
from ai.ga import train as run_ga
from ai.ga.train import _config_key, _seed_ga_randomness, ga_optimizer
from env.action_schema import ActionSchema


PROFILE = "current4"


def _make_cfg(index: int) -> dict:
    fraction = (index + 1) / 100.0
    return build_individual_config(
        turnover_rate=0.1,
        weights={
            name: fraction if index == 0 else 1.0 - fraction if index == 1 else 0.2
            for index, name in enumerate(ActionSchema().factor_names)
        },
    )


def _ga_cache(count: int) -> dict:
    return {
        str(index): {
            "individual_config": _make_cfg(index),
            "calmar": float(index),
        }
        for index in range(count)
    }


def _empty_state() -> dict:
    return {"population": [], "hall_of_fame": [], "score_cache": {}}


def test_returns_parents_plus_one_child_population():
    population = 20
    result = ga_optimizer(
        [], _empty_state(), population, population, PROFILE, _ga_cache(40)
    )
    assert len(result) == 2 * population


def test_true_elites_are_preserved():
    population = 20
    result = ga_optimizer(
        [], _empty_state(), population, population, PROFILE, _ga_cache(40)
    )
    elite_count = max(2, round(0.10 * population))
    expected = {_config_key(_make_cfg(index)) for index in range(40 - elite_count, 40)}
    assert expected <= {_config_key(config) for config in result}


def test_random_immigrants_are_requested(monkeypatch):
    population = 20
    requested = []

    def fake_generate(count, profile_name=None):
        requested.append((count, profile_name))
        return [_make_cfg(index + 60) for index in range(count)]

    monkeypatch.setattr(run_ga, "generate_initial_configs", fake_generate)
    ga_optimizer([], _empty_state(), population, population, PROFILE, _ga_cache(40))
    assert requested == [(3, PROFILE)]


def test_small_cache_falls_back_to_available_evaluated_parents():
    results = [
        {"individual_config": _make_cfg(index), "calmar": float(index)}
        for index in range(5)
    ]
    result = ga_optimizer(results, _empty_state(), 20, 20, PROFILE)
    assert len(result) == 25


def test_fixed_seed_reproduces_initial_population_and_breeding():
    _seed_ga_randomness(20260720)
    first_initial = generate_initial_configs(20)
    _seed_ga_randomness(20260720)
    second_initial = generate_initial_configs(20)
    assert [_config_key(config) for config in first_initial] == [
        _config_key(config) for config in second_initial
    ]

    cache = _ga_cache(40)
    _seed_ga_randomness(1234)
    first = ga_optimizer([], _empty_state(), 20, 20, PROFILE, cache)
    _seed_ga_randomness(1234)
    second = ga_optimizer([], _empty_state(), 20, 20, PROFILE, cache)
    assert [_config_key(config) for config in first] == [
        _config_key(config) for config in second
    ]


def test_breeding_searches_continuous_turnover_without_category_quantization():
    _seed_ga_randomness(412)
    result = ga_optimizer([], _empty_state(), 60, 60, PROFILE, _ga_cache(80))
    rates = [config["turnover_rate"] for config in result]
    assert all(0.0 <= rate <= 0.2 for rate in rates)
    assert any(rate * 50 != int(rate * 50) for rate in rates)
    assert len(set(rates)) > 10


def test_parent_index_preserves_sort_and_only_keys_new_candidates(monkeypatch):
    cache = _ga_cache(40)
    index = run_ga._TrainingParentPool(cache)
    calls = []
    original = run_ga._ga_cache_key
    monkeypatch.setattr(run_ga, "_ga_cache_key", lambda config: (calls.append(config), original(config))[1])
    assert index.update(cache) is index.rows
    assert not calls
    for position in (40, 41, 42):
        cache[str(position)] = {"individual_config": _make_cfg(position), "calmar": 30.0}
    actual = index.update(cache)
    assert len(calls) == 3
    expected = sorted(cache.values(), key=lambda entry: (entry["calmar"], original(entry["individual_config"])), reverse=True)
    assert [(row[0], row[1]) for row in actual] == [(row["individual_config"],row["calmar"]) for row in expected]


def test_parent_index_requires_one_append_only_cache():
    import pytest
    cache = _ga_cache(40)
    index = run_ga._TrainingParentPool(cache)
    with pytest.raises(ValueError, match="same append-only"):
        index.update(dict(cache))
    cache.pop(next(iter(cache)))
    with pytest.raises(ValueError, match="same append-only"):
        index.update(cache)
