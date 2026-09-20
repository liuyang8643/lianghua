"""Latest GA protocol: exact-contract cache continuation, never a legacy migration."""
from contextlib import ExitStack
import copy
import json
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from ai.ga import build_individual_config
from ai.ga.comparison import Comparison, config_sha, digest
from env.action_schema import ActionSchema


def write_json(path, payload):
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")


@pytest.fixture
def continuation_case(tmp_path, monkeypatch, resources):
    schema = ActionSchema()
    config = schema.to_static_config(schema.decode(np.zeros(schema.action_dim)))
    static = {key: value for key, value in config.items() if key != "factor_enabled"}
    config_file = tmp_path / "config.json"
    write_json(config_file, static)
    runtime = tmp_path / "runtime.npz"
    runtime.write_bytes(b"immutable local fixture")
    financial = {"snapshot_sha256": digest(runtime)}
    write_json(runtime.with_suffix(".manifest.json"), financial)
    monkeypatch.setattr("ai.ga.comparison.read_financial_snapshot_manifest", lambda _: financial)
    splits = {"train": ["2020-01-01", "2020-01-10"],
              "validation": ["2020-01-11", "2020-01-20"], "test": ["2020-01-21", "2020-01-31"]}
    split_file = tmp_path / "splits.json"
    write_json(split_file, splits)
    manifest = SimpleNamespace(source_sha256=digest(runtime), source_path=str(runtime),
        requested_start=splits["train"][0], requested_end=splits["train"][1],
        as_dict=lambda: {"source_sha256": digest(runtime)})
    episode = SimpleNamespace(factors=SimpleNamespace(schema_hash="2" * 64), prefilter_n=300,
        decision_start=0, decision_stop=2, runtime=SimpleNamespace(decision_start=0, decision_stop=2,
            manifest=manifest, decision_dates=np.array(["2020-01-02", "2020-01-10"], dtype="datetime64[D]")))
    args = SimpleNamespace(evaluation_splits=str(split_file), config=str(config_file), seed=71,
        population_size=32, generations=4, workers=1, eval_every_generations=2, continue_from=None, warm_start=None,
        lookback=64, runtime_path=str(runtime))
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    metrics = {"transition_count": 7, "calmar": 0.8}
    evaluate = lambda episode, config: {"metrics": metrics}
    original = Comparison(args, parent_dir, episode, evaluate, resources=resources)
    parent = original.identity
    rows = [{"generation": index, "config": config, "config_sha256": config_sha(config),
             "train": metrics, "scheduled_evaluation": index == 2, "elapsed_seconds": float(index)}
            for index in (1, 2)]
    rows[1].update(validation={"calmar": 0.4}, test={"calmar": 1000.0})
    history = {"identity_sha256": parent["sha256"], "rows": rows, "elapsed_seconds": 10.0,
               "opened": {"validation": True, "test": True},
               "baselines": {"train": metrics, "validation": metrics, "test": metrics}}
    write_json(parent_dir / "comparison.json", history)
    (parent_dir / "all_results.jsonl").write_text("\n".join(json.dumps({"generation": index, "config": config})
        for index in (0, 1)), encoding="utf-8")
    output = tmp_path / "continuation"
    output.mkdir()
    args.continue_from = str(parent_dir)
    return SimpleNamespace(args=args, output=output, episode=episode, evaluate=evaluate,
        parent=parent, parent_dir=parent_dir, history=history, config=config)


def test_same_contract_continuation_restores_only_observed_history_and_is_explicit_about_rng(continuation_case, resources):
    case = continuation_case
    def forbidden(*_):
        pytest.fail("same-contract continuation must reuse the recorded training baseline")
    comparison = Comparison(case.args, case.output, case.episode, forbidden, resources=resources)
    assert comparison.completed_generation == 2
    assert comparison.identity["continuation"]["start_generation"] == 3
    assert comparison.identity["continuation"]["optimizer_rng_restored"] is False
    assert comparison.identity["continuation"]["mode"] == "same_contract_training_cache_continuation_reseeded_breeder"
    assert comparison.selected["generation"] == 2 and "test" not in comparison.selected
    assert comparison.opened == {"validation": True, "test": True}
    assert comparison.holdout("test", case.config) == {"calmar": 1000.0}
    assert comparison.holdout("validation", case.config) == {"calmar": 0.4}


@pytest.mark.parametrize("change", ["version", "workers", "source_files", "eval_every_generations", "generations",
                                    "selection", "test_role", "holdout_scope", "evaluation_contract"])
def test_continuation_rejects_any_changed_protocol_even_if_parent_is_resigned(continuation_case, change, resources):
    case = continuation_case
    case.parent[change] = "changed signed protocol"
    case.parent.pop("sha256")
    case.parent["sha256"] = config_sha(case.parent)
    write_json(case.parent_dir / "comparison_identity.json", case.parent)
    with pytest.raises(ValueError, match=f"changed {change}"):
        Comparison(case.args, case.output, case.episode, case.evaluate, resources=resources)


@pytest.mark.parametrize("corruption", ["identity", "generation_gap", "cache_ahead", "cache_gap", "schedule", "config_sha"])
def test_continuation_rejects_incoherent_saved_history(continuation_case, corruption, resources):
    case = continuation_case
    if corruption == "identity":
        case.history["identity_sha256"] = "wrong"
    elif corruption == "generation_gap":
        case.history["rows"].pop(0)
    elif corruption == "cache_ahead":
        with (case.parent_dir / "all_results.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("\n" + json.dumps({"generation": 2, "config": case.config}))
    elif corruption == "cache_gap":
        (case.parent_dir / "all_results.jsonl").write_text(json.dumps({"generation": 1, "config": case.config}), encoding="utf-8")
    elif corruption == "schedule":
        case.history["rows"][0]["validation"] = {"calmar": 100}
    else:
        case.history["rows"][0]["config_sha256"] = "wrong"
    write_json(case.parent_dir / "comparison.json", case.history)
    with pytest.raises(ValueError):
        Comparison(case.args, case.output, case.episode, case.evaluate, resources=resources)


def test_tenth_and_final_only_and_test_never_selects(tmp_path):
    comparison = object.__new__(Comparison)
    comparison.root = tmp_path
    comparison.args = SimpleNamespace(eval_every_generations=10, generations=23)
    comparison.rows, comparison.selected = [], None
    comparison.started, comparison.parent_elapsed = 0.0, 0.0
    comparison.publish = lambda: None
    calls = []
    validation = iter((0.8, 0.7, 0.6))
    diagnostic_test = iter((-100.0, 100.0, 1000.0))
    def holdout(split, config):
        calls.append((comparison.completed_generation, split))
        return {'calmar': next(validation if split == 'validation' else diagnostic_test)}
    comparison.holdout = holdout
    best = {'individual_config': build_individual_config(turnover_rate=0.1), 'metrics': {'calmar': 0.4}}
    original = copy.deepcopy(best)
    for generation in range(23):
        comparison.evaluate(generation, best, generation + 1)
    assert calls == [(generation, split) for generation in (10, 20, 23) for split in ('validation', 'test')]
    assert len(comparison.rows) == 23
    assert [r['generation'] for r in comparison.rows if 'validation' in r] == [10, 20, 23]
    assert comparison.selected['generation'] == 10 and 'test' not in comparison.selected
    assert best == original



def test_holdout_residency_preserves_scalar_cache_baselines_and_validation_selection(tmp_path, monkeypatch, resources):
    comparison = object.__new__(Comparison)
    comparison.root = tmp_path
    comparison.args = SimpleNamespace(runtime_path="not-loaded.npz", lookback=504, eval_every_generations=1, generations=3)
    comparison.resources = resources
    train = SimpleNamespace(prefilter_n=300, factors=SimpleNamespace(schema_hash="same"))
    comparison.episodes = {"train": train}
    comparison.splits = {"validation": ("validation", "validation-end"), "test": ("test", "test-end")}
    comparison.cache, comparison.baselines = {}, {"train": {"calmar": 0.5}}
    comparison.baseline_config = {"id": "baseline"}
    comparison.rows, comparison.selected = [], None
    comparison.opened = {"validation": False, "test": False}
    comparison.started, comparison.parent_elapsed = 0.0, 0.0
    comparison.publish = lambda: None
    references, loads, evaluations = [], [], []

    class Episode:
        def __init__(self, split):
            self.split = split
            self.factors = SimpleNamespace(schema_hash="same")

    def prepare(_path, split, _end, **kwargs):
        assert kwargs == {"lookback": 504, "prefilter_n": 300, "encode_observations": False}
        episode = Episode(split)
        references.append(weakref.ref(episode))
        loads.append(split)
        return episode

    def evaluate(episode, config):
        key = config["id"]
        evaluations.append((episode.split, key))
        if key == "baseline":
            value = 0.3
        elif episode.split == "validation":
            value = {"A": 0.8, "B": 0.7, "C": 1.1}[key]
        else:
            assert comparison.selected["config"]["id"] == {"A": "A", "B": "A", "C": "C"}[key]
            value = {"A": -100.0, "B": 1000.0, "C": -1000.0}[key]
        return {"metrics": {"calmar": value}}

    class Resident:
        def __init__(self, prepared):
            self.episode = Episode(prepared.split)
        def __enter__(self):
            return self
        def __exit__(self, *_):
            self.episode = None
    monkeypatch.setattr("ai.ga.comparison.ResidentPreparedEpisode", Resident)
    monkeypatch.setattr("ai.ga.comparison.prepare_episode_from_runtime", prepare)
    comparison.evaluator = evaluate
    for generation, key in enumerate(("A", "B", "C")):
        candidate = {"individual_config": {"id": key}, "metrics": {"calmar": 0.6}}
        comparison.evaluate(generation, candidate, generation + 1)
        before = list(evaluations), list(loads), tuple(comparison.episodes)
        assert comparison.holdout("validation", {"id": key}) is comparison.rows[-1]["validation"]
        assert comparison.holdout("test", {"id": key}) is comparison.rows[-1]["test"]
        assert before == (evaluations, loads, tuple(comparison.episodes))
        assert candidate == {"individual_config": {"id": key}, "metrics": {"calmar": 0.6}}
    assert all(reference() is None for reference in references)
    assert loads == ["validation", "test"]
    assert evaluations.count(("validation", "baseline")) == evaluations.count(("test", "baseline")) == 1
    assert comparison.baselines == {"train": {"calmar": 0.5}, "validation": {"calmar": 0.3}, "test": {"calmar": 0.3}}
    assert len(comparison.cache) == 6
    assert comparison.selected["generation"] == 3 and "test" not in comparison.selected
    assert comparison.episodes["train"] is train


@pytest.fixture
def resources():
    with ExitStack() as stack:
        yield stack
