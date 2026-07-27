import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "strategy_opt_20260721"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_v17_holdout_state_matches_frozen_candidate_and_artifacts():
    manifest = json.loads((RESULTS / "v17_freeze_manifest.json").read_text())
    state = json.loads((RESULTS / "v17_holdout_state.json").read_text())
    assert state["candidate_sha256"] == manifest["candidate_sha256"]
    disclosure = state["project_level_holdout_disclosure"]
    assert disclosure["validation_is_project_level_pristine"] is False
    assert disclosure["test_was_project_level_pristine_before_v17_evaluation"] is True
    assert disclosure["combined_oos_is_independent_evidence"] is False
    assert not (
        ROOT / "configs/smallcap_v17_overnight_gap_w20_config.json"
    ).exists()
    rejection = state["execution_feasibility_rejection"]
    assert rejection["status"] == "rejected_not_deployable"
    assert "OvernightGapDown" in rejection["action"]

    for scope in ("validation", "test", "combined_oos"):
        result = state[scope]
        assert result["status"] == "evaluated_once_passed"
        assert result["passes_all_gates"] is True
        assert result["trade_open_audit_passes"] is True
        assert _sha256(RESULTS / result["diagnostics"]) == (
            result["diagnostics_sha256"]
        )
        assert _sha256(RESULTS / result["trade_open_audit"]) == (
            result["trade_open_audit_sha256"]
        )
        diagnostics = json.loads((RESULTS / result["diagnostics"]).read_text())
        audit = json.loads((RESULTS / result["trade_open_audit"]).read_text())
        assert diagnostics["passes_all_gates"] is True
        assert audit["passes"] is True
        assert diagnostics["full"]["calmar"] == pytest.approx(result["calmar"])


def test_v17_oos_cost_stress_hash_and_frozen_interpretation():
    state = json.loads((RESULTS / "v17_holdout_state.json").read_text())
    risk = state["execution_risk"]
    assert _sha256(RESULTS / risk["cost_stress"]) == risk["cost_stress_sha256"]
    stress = json.loads((RESULTS / risk["cost_stress"]).read_text())
    scenarios = {row["extra_one_way_bps"]: row for row in stress["scenarios"]}
    assert scenarios[5]["calmar"] == pytest.approx(
        risk["extra_5bps_per_side_calmar"]
    )
    assert scenarios[10]["calmar"] == pytest.approx(
        risk["extra_10bps_per_side_calmar"]
    )
    assert scenarios[5]["calmar"] >= 2.5
    assert scenarios[10]["calmar"] < 2.5
