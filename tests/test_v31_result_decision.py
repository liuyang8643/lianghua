import hashlib
import json
from pathlib import Path


ROOT = Path("results/strategy_opt_20260721")
OUTPUT = ROOT / "v31_natural_calendar_semantic_correction"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_v31_preregistered_semantic_correction_is_closed_and_rejected():
    decision = json.loads(
        (ROOT / "v31_semantic_correction_training_decision.json")
        .read_text(encoding="utf-8")
    )
    rows = _rows(OUTPUT / "all_results.jsonl")
    assert len(rows) == 648
    assert len({json.dumps(row["individual_config"], sort_keys=True) for row in rows}) == 648
    assert sum(row["full_calmar"] >= 2.5 for row in rows) == 82
    assert sum(row["worst_fold_calmar"] >= 1.0 for row in rows) == 2
    assert sum(row["worst_fold_calmar"] >= 1.1 for row in rows) == 0
    assert sum(row["worst_fold_calmar"] >= 1.5 for row in rows) == 0
    assert decision["status"] == "all_648_candidates_rejected_at_natural_core_gate"
    assert decision["audit_summary"]["core_eligible_count"] == 0
    assert decision["strategy_holdout_available_to_factor_or_worker"] is False
    assert decision["strategy_holdout_evaluated"] is False


def test_v31_metadata_exposes_loader_buffer_and_pool_denominator():
    metadata = json.loads((OUTPUT / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["runtime_loader_returned_post_train_rows"] == 7
    assert metadata["runtime_post_train_rows_discarded_before_factor"] == 7
    assert metadata["strategy_holdout_available_to_factor_or_worker"] is False
    assert metadata["strategy_holdout_evaluated"] is False
    assert metadata["ranking_stock_pool_prefixes"] == ["60", "00", "30"]
    assert metadata["ranking_pool_column_count"] == 4737
    assert metadata["runtime_column_count"] == 5350
    assert metadata["ranking_outside_pool_column_count"] == 613


def test_v31_matches_invalidated_v30_only_as_a_diagnostic():
    old = _rows(ROOT / "v30_natural_calendar_rescore/all_results.jsonl")
    new = _rows(OUTPUT / "all_results.jsonl")
    metric_fields = (
        "full_calmar",
        "worst_fold_calmar",
        "robust_score",
        "average_exposure",
        "annualized",
        "max_drawdown",
        "sharpe",
        "total_return",
    )
    for old_row, new_row in zip(old, new):
        assert old_row["candidate_index"] == new_row["candidate_index"]
        for field in metric_fields:
            assert old_row[field] == new_row[field]
        assert old_row["fold_calmars"] == new_row["fold_calmars"]


def test_v31_artifact_hashes_and_preregistration_order():
    decision = json.loads(
        (ROOT / "v31_semantic_correction_training_decision.json")
        .read_text(encoding="utf-8")
    )
    for payload in decision["artifacts"].values():
        path = ROOT / payload["path"]
        assert path.stat().st_size == payload["bytes"]
        assert _sha256(path) == payload["sha256"]
    plan = ROOT / "v31_semantic_correction_preregistered_plan.json"
    assert plan.stat().st_mtime < (OUTPUT / "all_results.jsonl").stat().st_mtime
    assert not (OUTPUT / "holdout_diagnostics.json").exists()
    for row in _rows(OUTPUT / "all_results.jsonl"):
        assert not any(key.startswith("val_") or key.startswith("test_") for key in row)
