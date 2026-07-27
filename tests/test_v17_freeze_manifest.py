import hashlib
import json
from pathlib import Path

import pytest

from research_holdout_diagnostics import GATES, PERIODS


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "strategy_opt_20260721"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_v17_frozen_config_and_training_selection_have_not_drifted():
    manifest = json.loads((RESULTS / "v17_freeze_manifest.json").read_text())
    frozen_path = RESULTS / manifest["candidate"]
    source_path = RESULTS / manifest["source_config"]
    assert json.loads(frozen_path.read_text()) == json.loads(source_path.read_text())
    assert _sha256(frozen_path) == manifest["candidate_sha256"]
    assert _sha256(
        RESULTS / manifest["source_training_run"] / "record.json"
    ) == manifest["source_record_sha256"]
    assert _sha256(ROOT / "factor_db/factors/OvernightGapDown.py") == (
        manifest["factor_sha256"]
    )

    comparison = json.loads(
        (RESULTS / "v17_overnight_gap_down_comparison.json").read_text()
    )
    selected = next(
        row for row in comparison["candidates"]
        if row["run"].replace("\\", "/").endswith(
            "v17_overnight_gap_down_w20_train"
        )
    )
    frozen = manifest["training_selection"]
    assert selected["full"]["calmar"] == pytest.approx(frozen["full_calmar"])
    assert selected["worst_calendar_block_calmar"] == pytest.approx(
        frozen["worst_calendar_3y_calmar"]
    )
    assert selected["rolling_3y"]["min_calmar"] == pytest.approx(
        frozen["rolling_3y_min_calmar"]
    )
    assert selected["rolling_3y"]["p10_calmar"] == pytest.approx(
        frozen["rolling_3y_p10_calmar"]
    )
    assert selected["return_concentration"][
        "largest_year_log_wealth_share"
    ] == pytest.approx(frozen["largest_year_log_wealth_share"])
    assert selected["execution"]["average_exposure"] == pytest.approx(
        frozen["average_exposure"]
    )
    assert selected["passes_all_gates"] is True
    assert sum(row["passes_all_gates"] for row in comparison["candidates"]) == 1


def test_v17_training_artifact_hashes_have_not_drifted():
    manifest = json.loads((RESULTS / "v17_freeze_manifest.json").read_text())
    for artifact in manifest["training_artifacts"].values():
        assert _sha256(RESULTS / artifact["path"]) == artifact["sha256"]


def test_v17_holdout_periods_and_gates_were_frozen_with_the_code():
    manifest = json.loads((RESULTS / "v17_freeze_manifest.json").read_text())
    policy = manifest["holdout_policy"]
    for scope, (first, last) in PERIODS.items():
        assert policy[scope]["period"] == [first, last]
        for gate_name, expected in GATES[scope].items():
            assert policy[scope][gate_name] == expected

    implementations = manifest["metric_implementation"]
    assert _sha256(ROOT / "research_train_robustness.py") == (
        implementations["training_sha256"]
    )
    assert _sha256(ROOT / "research_holdout_diagnostics.py") == (
        implementations["holdout_sha256"]
    )
    assert _sha256(ROOT / "research_holdout_trade_open_audit.py") == (
        implementations["holdout_trade_open_audit_sha256"]
    )
    assert policy["validation"]["status"] == "sealed_unread"
    assert policy["test"]["status"] == "sealed_unread_until_validation_passes"
