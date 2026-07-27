import hashlib
import json
from pathlib import Path

import pytest

from research_holdout_diagnostics import GATES, PERIODS


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "strategy_opt_20260721"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_v12_frozen_config_and_training_metrics_have_not_drifted():
    manifest = json.loads((RESULTS / "v12_freeze_manifest.json").read_text())
    frozen_path = RESULTS / manifest["candidate"]
    source_path = RESULTS / "v12_amount20_max55m_config.json"
    assert json.loads(frozen_path.read_text()) == json.loads(source_path.read_text())
    assert _sha256(frozen_path) == manifest["candidate_sha256"]
    assert _sha256(
        RESULTS / manifest["source_training_run"] / "record.json"
    ) == manifest["source_record_sha256"]

    artifact = json.loads(
        (RESULTS / "v12_amount20_ceiling_dense_neighborhood.json").read_text()
    )
    selected = next(
        row for row in artifact["candidates"]
        if row["run"].replace("\\", "/").endswith("v12_amount20_max55m_train")
    )
    frozen_metrics = manifest["training_selection"]
    assert selected["full"]["calmar"] == pytest.approx(
        frozen_metrics["full_calmar"]
    )
    assert selected["worst_calendar_block_calmar"] == pytest.approx(
        frozen_metrics["worst_calendar_3y_calmar"]
    )
    assert selected["rolling_3y"]["p10_calmar"] == pytest.approx(
        frozen_metrics["rolling_3y_p10_calmar"]
    )
    assert selected["return_concentration"][
        "largest_year_log_wealth_share"
    ] == pytest.approx(frozen_metrics["largest_year_log_wealth_share"])
    assert selected["execution"]["average_exposure"] == pytest.approx(
        frozen_metrics["average_exposure"]
    )
    assert selected["passes_all_gates"] is True


def test_v12_holdout_periods_and_gates_were_frozen_with_the_code():
    manifest = json.loads((RESULTS / "v12_freeze_manifest.json").read_text())
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
