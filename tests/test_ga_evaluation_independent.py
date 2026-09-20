"""Independent explicit-GA split and holdout-isolation acceptance checks."""
from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai.ga import build_individual_config
from ai.ga.comparison import Comparison, read_evaluation_splits
from ai.ga.train import _evaluate_individual
from env.action_schema import ActionSchema
from env.backtest import prepare_episode_from_runtime
from test_backtest_lightweight import write_canonical_runtime


SPLITS = {"train": ["2020-06-01", "2020-06-08"],
          "validation": ["2020-06-09", "2020-06-13"],
          "test": ["2020-06-14", "2020-06-18"]}


@pytest.mark.parametrize("change", [
    ("train", ["2020-06-08", "2020-06-01"]),
    ("train", ["2020-06-01", "2020-06-01"]),
    ("validation", ["2020-06-08", "2020-06-13"]),
    ("test", ["2020-06-13", "2020-06-18"]),
    ("validation", ["2020-6-09", "2020-06-13"]),
    ("validation", ["2020-02-30", "2020-06-13"]),
    ("test", None),
    ("test", ["2020-06-14"]),
])
def test_split_boundaries_fail_before_any_runtime_read(tmp_path, change):
    splits = copy.deepcopy(SPLITS)
    splits[change[0]] = change[1]
    path = tmp_path / "splits.json"
    path.write_text(json.dumps(splits), encoding="utf-8")
    with pytest.raises(ValueError):
        read_evaluation_splits(path)


def test_split_file_identity_tracks_exact_bytes(tmp_path):
    path = tmp_path / "splits.json"
    path.write_text(json.dumps(SPLITS), encoding="utf-8")
    splits, sha = read_evaluation_splits(path)
    assert splits == SPLITS
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text(json.dumps(SPLITS, indent=2), encoding="utf-8")
    same_splits, other_sha = read_evaluation_splits(path)
    assert same_splits == splits and sha != other_sha


@pytest.fixture
def explicit_case(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime.npz"
    write_canonical_runtime(runtime, stocks=30)
    synthetic_manifest = {"synthetic_test_fixture": True,
                          "snapshot_sha256": hashlib.sha256(runtime.read_bytes()).hexdigest()}
    runtime.with_suffix(".manifest.json").write_text(json.dumps(synthetic_manifest), encoding="utf-8")

    def synthetic_verifier(path):
        assert Path(path) == runtime
        return synthetic_manifest

    monkeypatch.setattr("ai.ga.comparison.read_financial_snapshot_manifest", synthetic_verifier)
    config = tmp_path / "config.json"
    static_config = build_individual_config(turnover_rate=0.1)
    static_config.pop("factor_enabled")
    config.write_text(json.dumps({**static_config, "prefilter_n": 300}), encoding="utf-8")
    split_file = tmp_path / "splits.json"
    split_file.write_text(json.dumps(SPLITS), encoding="utf-8")
    episode = prepare_episode_from_runtime(runtime, *SPLITS["train"], lookback=64,
                                           prefilter_n=300, encode_observations=False)
    output = tmp_path / "run"
    output.mkdir()
    args = SimpleNamespace(evaluation_splits=str(split_file),
        eval_every_generations=10, continue_from=None, warm_start=None, config=str(config),
        runtime_path=str(runtime), lookback=64, seed=20260827,
        population_size=32, generations=23, workers=1)
    return args, output, episode


def test_standalone_actual_synthetic_accounts_keep_holdouts_out_of_selection(explicit_case, resources):
    args, output, episode = explicit_case
    calls = []

    def evaluate(actual_episode, config):
        calls.append(str(actual_episode.runtime.decision_dates[0]))
        return _evaluate_individual(actual_episode, config)

    comparison = Comparison(args, output, episode, evaluate, resources=resources)
    assert calls == [SPLITS["train"][0]]
    assert comparison.opened == {"validation": False, "test": False}
    assert "ppo_reference_identity" not in comparison.identity
    assert "ppo_contract" not in comparison.identity
    assert comparison.identity["evaluation_contract"]["splits"] == SPLITS
    assert len(comparison.identity["genes"]) == 12
    assert comparison.identity["test_role"] == "repeated_diagnostic_only_never_selection"
    assert comparison.identity["deployment_bundle"] is False
    baseline_calls = len(calls)
    schema = ActionSchema()
    for generation in range(23):
        candidate = {"individual_config": build_individual_config(turnover_rate=0.1,
            weights=dict.fromkeys(schema.factor_names, 0.1 + generation / 100)),
            "metrics": {"calmar": 1.0}}
        original = copy.deepcopy(candidate)
        comparison.evaluate(generation, candidate, generation + 1)
        assert candidate == original
        if generation < 9:
            assert len(calls) == baseline_calls
            assert comparison.opened == {"validation": False, "test": False}
        assert len({"validation", "test"}.intersection(comparison.episodes)) <= 2
    comparison.finish()
    assert comparison.phase == "complete"
    assert [r["generation"] for r in comparison.rows if "validation" in r] == [10, 20, 23]
    assert [r["generation"] for r in comparison.rows if "test" in r] == [10, 20, 23]
    selected = json.loads((output / "validation_selected.json").read_text("utf-8"))
    assert "test" not in selected
    expected = max((row for row in comparison.rows if "validation" in row),
                   key=lambda row: row["validation"]["calmar"])
    assert selected["config_sha256"] == expected["config_sha256"]


def test_standalone_rejects_train_range_mismatch_before_baseline(explicit_case, resources):
    args, output, episode = explicit_case
    changed = copy.deepcopy(SPLITS)
    changed["train"][0] = "2020-05-31"
    Path(args.evaluation_splits).write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="training requested range"):
        Comparison(args, output, episode, lambda *_: pytest.fail("baseline ran before range check"), resources=resources)


def test_standalone_manifest_rejection_happens_before_baseline(explicit_case, monkeypatch, resources):
    args, output, episode = explicit_case

    def rejected(_path):
        raise ValueError("synthetic financial manifest rejected")

    monkeypatch.setattr("ai.ga.comparison.read_financial_snapshot_manifest", rejected)
    with pytest.raises(ValueError, match="financial manifest rejected"):
        Comparison(args, output, episode, lambda *_: pytest.fail("unverified financial snapshot evaluated"), resources=resources)
    assert not (output / "comparison_identity.json").exists()


def test_failed_evaluation_publishes_terminal_state(explicit_case, resources):
    args, output, episode = explicit_case
    comparison = Comparison(args, output, episode, _evaluate_individual, resources=resources)
    comparison.phase = 'preparing_test'
    comparison.fail(MemoryError('allocation failed'))
    saved = json.loads((output / 'comparison.json').read_text(encoding='utf8'))
    assert saved['state'] == 'failed'
    assert saved['failure'] == {'type': 'MemoryError', 'message': 'allocation failed'}


@pytest.fixture
def resources():
    with ExitStack() as stack:
        yield stack
