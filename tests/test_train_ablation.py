import json

import pytest

from testback.analyze_train_ablation import compare_training_runs


def _write_run(path, *, seed=7, profile="baseline", fitness=1.0, holdout_value=None):
    path.mkdir()
    (path / "run_metadata.json").write_text(json.dumps({
        "seed": seed, "profile": profile, "sealed_holdout": True,
        "training_objective": {
            "mode": "robust_calmar", "folds": 3,
            "full_weight": 0.5, "min_average_exposure": 0.45,
        },
    }), encoding="utf-8")
    (path / "holdout_diagnostics.json").write_text("{}", encoding="utf-8")
    row = {
        "fitness": fitness, "calmar": fitness + 0.5, "sharpe": fitness + 1.0,
        "fold_calmars": [fitness, fitness + 1.0, fitness + 2.0],
        "val_calmar": holdout_value, "config": {"weights": {"Base": 1.0}},
    }
    (path / "all_results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_compare_training_runs_uses_only_training_fitness(tmp_path):
    baseline = tmp_path / "baseline"
    experiment = tmp_path / "experiment"
    _write_run(baseline, fitness=1.0)
    _write_run(experiment, profile="experiment", fitness=1.2)

    result = compare_training_runs(baseline, experiment)

    assert result["selection_scope"] == "train_only"
    assert result["training_winner"] == "experiment"
    assert result["delta"]["fitness"] == pytest.approx(0.2)
    assert result["delta"]["fold_calmars"] == pytest.approx([0.2, 0.2, 0.2])
    assert sum(result["baseline"]["normalized_weights"].values()) == pytest.approx(1.0)


def test_compare_training_runs_rejects_holdout_metrics(tmp_path):
    baseline = tmp_path / "baseline"
    experiment = tmp_path / "experiment"
    _write_run(baseline)
    _write_run(experiment, holdout_value=9.0)

    with pytest.raises(ValueError, match="holdout"):
        compare_training_runs(baseline, experiment)


def test_compare_training_runs_requires_matching_seed(tmp_path):
    baseline = tmp_path / "baseline"
    experiment = tmp_path / "experiment"
    _write_run(baseline, seed=7)
    _write_run(experiment, seed=8)

    with pytest.raises(ValueError, match="随机种子不一致"):
        compare_training_runs(baseline, experiment)


def test_compare_training_runs_requires_matching_objective(tmp_path):
    baseline = tmp_path / "baseline"
    experiment = tmp_path / "experiment"
    _write_run(baseline)
    _write_run(experiment)
    metadata_path = experiment / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["training_objective"]["min_average_exposure"] = 0.2
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="training_objective 不一致"):
        compare_training_runs(baseline, experiment)
