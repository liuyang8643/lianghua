"""Small synthetic GA search: no real dataset, PPO model or PPO learner."""
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai.ga import build_individual_config
from ai.ga import comparison as comparison_module
from ai.ga import train as ga_train
from env.backtest import ENVIRONMENT_SCHEMA_VERSION
from test_backtest_lightweight import write_canonical_runtime
from utils.atomic_file import file_sha256


@pytest.mark.parametrize('mode', ['new', 'resume_identity', 'resume_cache'])
def test_rejected_ga_start_preserves_all_existing_run_bytes(tmp_path, monkeypatch, mode):
    output = tmp_path / 'existing'
    output.mkdir()
    (output / 'training_report.json').write_bytes(b'{"status":"complete","sentinel":17}')
    (output / 'run_metadata.json').write_text(json.dumps({'identity': 'old'}), encoding='utf8')
    (output / 'all_results.jsonl').write_bytes(b'corrupted-cache\n')
    (output / 'ga.log').write_bytes(b'existing audit log\n')
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    args = ga_train.parse_args(['--mode', 'debug', '--output-dir', str(output),
                               '--runtime', str(tmp_path / 'unused.npz')])
    monkeypatch.setattr(ga_train, 'prepare_episode_from_runtime', lambda *_a, **_k: object())
    monkeypatch.setattr(ga_train, '_canonical_ga_metadata',
                        lambda **_: {'identity': 'old' if mode == 'resume_cache' else 'new'})
    with pytest.raises((FileExistsError, ValueError)):
        ga_train._run_ga(args, {'population_size': 2, 'generations': 2},
                         [datetime(2020, 6, 1), datetime(2020, 6, 8)],
                         resume_dir=None if mode == 'new' else output)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


def test_ga_and_ppo_default_evaluation_cadence_is_shared():
    from configs.training import DEFAULT_EVALUATION_EVERY
    from ai.rl.train import build_parser
    assert DEFAULT_EVALUATION_EVERY == 50
    assert ga_train.parse_args([]).eval_every_generations == DEFAULT_EVALUATION_EVERY
    assert build_parser().parse_args(['--runtime', 'unused.npz']).eval_every_rollouts == DEFAULT_EVALUATION_EVERY


@pytest.mark.parametrize("continue_after_first_generation,warm_start", [(False, False), (True, False), (False, True)])
def test_standalone_two_generation_ga_resolves_identity_and_evaluates_three_splits(tmp_path, monkeypatch, continue_after_first_generation, warm_start):
    runtime = tmp_path / "runtime.npz"
    write_canonical_runtime(runtime, stocks=30)
    financial = {"snapshot_sha256": file_sha256(runtime), "manifest_sha256": "a" * 64,
                 "source": "synthetic test fixture only"}
    runtime.with_suffix(".manifest.json").write_text(json.dumps(financial), encoding="utf8")
    monkeypatch.setattr(comparison_module, "read_financial_snapshot_manifest", lambda path: financial)
    monkeypatch.setattr(ga_train, "latest_runtime_npz_path", lambda: runtime)
    monkeypatch.setattr(ga_train.os, "cpu_count", lambda: 1)
    splits = {"train": ["2020-06-01", "2020-06-08"],
              "validation": ["2020-06-09", "2020-06-13"],
              "test": ["2020-06-14", "2020-06-18"]}
    split_file = tmp_path / "splits.json"
    split_file.write_text(json.dumps(splits), encoding="utf8")
    static = build_individual_config(turnover_rate=0.1)
    static.pop("factor_enabled")
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({**static, "prefilter_n": 300}), encoding="utf8")
    output = tmp_path / "run"
    args = SimpleNamespace(mode="ga", output_dir=str(output), runtime_path=None, config=str(config_file),
        lookback=64, workers=None, seed=17, candidate_configs=None, warm_start=None,
        evaluation_splits=str(split_file), continue_from=None,
        eval_every_generations=10, population_size=None, generations=None)
    dates = [datetime.fromisoformat(value) for value in splits["train"]]
    parent_cache = None
    if warm_start:
        candidate_file = tmp_path / 'inherited.json'
        inherited_config = build_individual_config(turnover_rate=0.1)
        candidate_file.write_text(json.dumps({'configs': [inherited_config]}), encoding='utf8')
        args.warm_start = str(candidate_file)
    if continue_after_first_generation:
        original_optimizer = ga_train.ga_optimizer
        class StopAfterCompleteGeneration(RuntimeError):
            pass
        def stop_after_generation(*args, **kwargs):
            raise StopAfterCompleteGeneration
        monkeypatch.setattr(ga_train, "ga_optimizer", stop_after_generation)
        with pytest.raises(StopAfterCompleteGeneration):
            ga_train._run_ga(args, {"population_size": 2, "generations": 2}, dates)
        monkeypatch.setattr(ga_train, "ga_optimizer", original_optimizer)
        parent_dir = output
        parent_cache = (parent_dir / "all_results.jsonl").read_bytes()
        args.continue_from = str(parent_dir)
        output = tmp_path / "continued"
        args.output_dir = str(output)
    winner = ga_train._run_ga(args, {"population_size": 2, "generations": 2}, dates)
    assert winner["full_investment_contract_satisfied"] is True
    identity = json.loads((output / "comparison_identity.json").read_text("utf8"))
    assert identity["version"].endswith("v9-canonical-continuous-actions")
    assert (identity["population"], identity["generations"], identity["workers"]) == (2, 2, 20)
    assert "ppo_contract" not in identity and "ppo_reference_identity" not in identity
    contract = identity["evaluation_contract"]
    assert contract["splits"] == splits and contract["lookback"] == 64
    assert contract["split_file"]["sha256"] == file_sha256(split_file)
    assert contract["runtime"]["file_sha256"] == financial["snapshot_sha256"]
    assert contract["environment"]["schema_version"] == ENVIRONMENT_SCHEMA_VERSION
    assert Path(args.runtime_path) == runtime.resolve()
    report = json.loads((output / "comparison.json").read_text("utf8"))
    assert report["state"] == "complete" and report["completed_generation"] == 2
    assert report["opened"] == {"validation": True, "test": True}
    assert set(report["baselines"]) == {"train", "validation", "test"}
    assert "validation" not in report["rows"][0] and "test" not in report["rows"][0]
    assert {"validation", "test"}.issubset(report["rows"][1])
    assert report["selected"]["generation"] == 2 and "test" not in report["selected"]
    if warm_start:
        initialization = identity['initialization']
        assert initialization['mode'] == 'candidate_warm_start_new_root'
        assert initialization['candidate_file'] == str(candidate_file.resolve())
        assert initialization['candidate_file_sha256'] == file_sha256(candidate_file)
        assert initialization['training_cache_reused'] is False
        assert initialization['holdout_metrics_reused'] is False
        assert 'continuation' not in identity
        results = [json.loads(line) for line in (output / 'all_results.jsonl').read_text('utf8').splitlines()]
        initial = [row for row in results if row['generation'] == 0]
        assert len(initial) == 1
        assert initial[0]['config'] == ga_train._canonical_ga_config(inherited_config, ga_train.DEFAULT_GA_PROFILE)
        assert initial[0]['calmar'] == report['rows'][0]['train']['calmar']
    if parent_cache is not None:
        assert (output / "all_results.jsonl").read_bytes().startswith(parent_cache)
        assert (parent_dir / "all_results.jsonl").read_bytes() == parent_cache
        assert identity["continuation"]["start_generation"] == 2
        assert identity["continuation"]["optimizer_rng_restored"] is False


@pytest.mark.parametrize('conflict', ['--resume', '--continue-from'])
def test_cli_rejects_mixed_warm_start_and_continuation(conflict):
    with pytest.raises(SystemExit):
        ga_train.parse_args(['--warm-start', 'candidates.json', conflict, 'parent'])


@pytest.mark.parametrize('resume', [False, True])
def test_runtime_rejects_mixed_warm_start_before_loading_or_writing(tmp_path, resume):
    args = SimpleNamespace(warm_start='candidates.json', continue_from=None if resume else 'parent')
    with pytest.raises(ValueError, match='cannot be combined'):
        ga_train._run_ga(args, {}, [], resume_dir=tmp_path if resume else None)
    assert list(tmp_path.iterdir()) == []


def test_cli_reference_and_explicit_splits_are_mutually_exclusive(monkeypatch):
    monkeypatch.setattr("sys.argv", ["ga", "--ppo-reference", "unused", "--evaluation-splits", "unused"])
    with pytest.raises(SystemExit) as stopped:
        ga_train.main()
    assert stopped.value.code == 2
