"""Compare two completed sealed GA runs using training metrics only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


_HOLDOUT_FIELDS = (
    "val_calmar", "val_sharpe", "val_annualized", "val_max_drawdown",
    "test_calmar", "test_sharpe", "test_annualized", "test_max_drawdown",
)


def _load_training_run(directory: Path) -> tuple[dict, list[dict]]:
    metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
    if not metadata.get("sealed_holdout"):
        raise ValueError(f"{directory} 不是 sealed-holdout 运行")
    if not (directory / "holdout_diagnostics.json").is_file():
        raise ValueError(f"{directory} 尚未完成并冻结训练候选")

    rows = [
        json.loads(line)
        for line in (directory / "all_results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"{directory} 没有训练结果")
    leaked = [
        field for field in _HOLDOUT_FIELDS
        if any(row.get(field) is not None for row in rows)
    ]
    if leaked:
        raise ValueError(f"训练记录混入 holdout 字段: {', '.join(leaked)}")
    return metadata, rows


def _training_candidate(rows: list[dict]) -> dict:
    return max(rows, key=lambda row: row["fitness"])


def _normalized_weights(config: dict) -> dict[str, float]:
    weights = config["weights"]
    total = sum(abs(float(value)) for value in weights.values())
    return {
        name: (float(value) / total if total else 0.0)
        for name, value in weights.items()
    }


def compare_training_runs(baseline_dir: Path, experiment_dir: Path) -> dict:
    baseline_meta, baseline_rows = _load_training_run(baseline_dir)
    experiment_meta, experiment_rows = _load_training_run(experiment_dir)
    if baseline_meta.get("seed") != experiment_meta.get("seed"):
        raise ValueError(
            f"随机种子不一致: {baseline_meta.get('seed')} != {experiment_meta.get('seed')}"
        )
    if baseline_meta.get("training_objective") != experiment_meta.get("training_objective"):
        raise ValueError("training_objective 不一致")

    baseline = _training_candidate(baseline_rows)
    experiment = _training_candidate(experiment_rows)
    baseline_folds = baseline.get("fold_calmars", [])
    experiment_folds = experiment.get("fold_calmars", [])
    if len(baseline_folds) != len(experiment_folds):
        raise ValueError("训练折数量不一致")

    baseline_weights = baseline["config"]["weights"]
    experiment_weights = experiment["config"]["weights"]
    factor_changes = {
        "added": sorted(set(experiment_weights) - set(baseline_weights)),
        "removed": sorted(set(baseline_weights) - set(experiment_weights)),
    }
    winner = "experiment" if experiment["fitness"] > baseline["fitness"] else "baseline"
    return {
        "selection_scope": "train_only",
        "seed": baseline_meta["seed"],
        "training_objective": baseline_meta.get("training_objective"),
        "baseline_profile": baseline_meta.get("profile"),
        "experiment_profile": experiment_meta.get("profile"),
        "factor_changes": factor_changes,
        "baseline": {
            "fitness": baseline["fitness"], "calmar": baseline["calmar"],
            "sharpe": baseline["sharpe"], "fold_calmars": baseline_folds,
            "average_exposure": baseline.get("average_exposure"),
            "normalized_weights": _normalized_weights(baseline["config"]),
            "config": baseline["config"],
        },
        "experiment": {
            "fitness": experiment["fitness"], "calmar": experiment["calmar"],
            "sharpe": experiment["sharpe"], "fold_calmars": experiment_folds,
            "average_exposure": experiment.get("average_exposure"),
            "normalized_weights": _normalized_weights(experiment["config"]),
            "config": experiment["config"],
        },
        "delta": {
            "fitness": experiment["fitness"] - baseline["fitness"],
            "calmar": experiment["calmar"] - baseline["calmar"],
            "sharpe": experiment["sharpe"] - baseline["sharpe"],
            "average_exposure": (
                experiment.get("average_exposure", 0.0)
                - baseline.get("average_exposure", 0.0)
            ),
            "fold_calmars": [
                right - left for left, right in zip(baseline_folds, experiment_folds)
            ],
        },
        "training_winner": winner,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="训练集限定的 sealed GA 消融比较")
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = compare_training_runs(args.baseline_dir, args.experiment_dir)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
