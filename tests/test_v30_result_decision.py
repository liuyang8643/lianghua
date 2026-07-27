import hashlib
import json
from pathlib import Path


ROOT = Path("results/strategy_opt_20260721")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_v30_decision_recomputes_closed_grid_rejection():
    decision = json.loads(
        (ROOT / "v30_natural_calendar_rescore_training_decision.json")
        .read_text(encoding="utf-8")
    )
    rows = [
        json.loads(line)
        for line in (ROOT / "v30_natural_calendar_rescore/all_results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 648
    assert len({json.dumps(row["individual_config"], sort_keys=True) for row in rows}) == 648
    assert sum(row["full_calmar"] >= 2.5 for row in rows) == 82
    assert sum(row["worst_fold_calmar"] >= 1.0 for row in rows) == 2
    assert sum(row["worst_fold_calmar"] >= 1.1 for row in rows) == 0
    assert sum(row["worst_fold_calmar"] >= 1.5 for row in rows) == 0
    assert decision["status"] == "invalidated_after_independent_semantic_audit"
    assert decision["audit_summary"]["core_eligible_count"] == 0
    assert decision["runtime_loader_returned_post_train_rows"] == 7
    assert decision["strategy_holdout_evaluated"] is False


def test_v30_artifact_hashes_and_preregistration_order():
    decision = json.loads(
        (ROOT / "v30_natural_calendar_rescore_training_decision.json")
        .read_text(encoding="utf-8")
    )
    for payload in decision["artifacts"].values():
        path = ROOT / payload["path"]
        assert path.stat().st_size == payload["bytes"]
        assert _sha256(path) == payload["sha256"]
    plan = ROOT / "v30_natural_calendar_rescore_preregistered_plan.json"
    results = ROOT / "v30_natural_calendar_rescore/all_results.jsonl"
    assert plan.stat().st_mtime < results.stat().st_mtime


def test_v30_output_contains_no_holdout_fields_or_artifacts():
    output = ROOT / "v30_natural_calendar_rescore"
    metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["train_first_date"] == "2010-01-04"
    assert metadata["train_last_date"] == "2018-12-28"
    assert metadata["train_days"] == 2187
    assert metadata["strategy_holdout_loaded"] is False
    assert metadata["strategy_holdout_evaluated"] is False
    assert not (output / "holdout_diagnostics.json").exists()
    for line in (output / "all_results.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert not any(key.startswith("val_") or key.startswith("test_") for key in row)
