import pytest

from ai.bundle import (
    BUNDLE_VERSION, DEPLOYMENT_GATE_NAMES, BundleManifest,
    policy_source_sha256,
)


def test_policy_identity_binds_shared_numeric_execution_kernel(tmp_path):
    shared = tmp_path / "utils" / "stable_sort.py"
    shared.parent.mkdir()
    shared.write_text("def stable_radix_order():\n    return 1\n", encoding="utf-8")
    before = policy_source_sha256(tmp_path)
    shared.write_text("def stable_radix_order():\n    return 2\n", encoding="utf-8")
    assert policy_source_sha256(tmp_path) != before


def _summary(calmar=1.0):
    return {
        "full_investment_contract_satisfied": True,
        "metrics": {"calmar": calmar},
    }


def _manifest():
    convergence = {name: True for name in DEPLOYMENT_GATE_NAMES}
    convergence["technical_convergence"] = True
    return BundleManifest(
        created_at="2026-08-30T00:00:00+00:00",
        algorithm="stable_baselines3.PPO",
        model_file="model.zip",
        model_sha256="a" * 64,
        normalizer_file="normalizer.json",
        normalizer_sha256="b" * 64,
        config_file="strategy_config.json",
        config_sha256="c" * 64,
        source_sha256="d" * 64,
        runtime={
            "financial_snapshot": {
                "manifest_sha256": "a" * 64,
                "snapshot_sha256": "b" * 64,
                "financial_identity_sha256": "c" * 64,
                "panel_builder_version": "fixture-panel",
                "financial_replay_version": "fixture-replay",
                "availability": "announcement_date < decision_date",
                "pit_evidence_limit": "test fixture",
            },
            "lineage": {
                "lineage_version": "v1",
                "runtime_schema_hash": "schema",
                "generation_semantics_sha256": "generation",
                "semantic_sha256": "semantic",
                "prefix_start": "2010-01-01",
                "prefix_end": "2026-08-28",
                "prefix_rows": 1,
                "stock_vocabulary_sha256": "stocks",
                "date_axis_sha256": "dates",
                "fields": [],
            }
        },
        factors={"schema_hash": "factors"},
        observation_schema={"schema_version": "observation"},
        encoded_schema={"identifier": "encoded"},
        action_schema={"schema_version": "action"},
        environment={"schema_version": "environment"},
        training={
            "convergence": convergence,
            "checkpoint_selection": {"objective": "full_validation_calmar"},
        },
        evaluation={
            "train": _summary(2.0),
            "validation": _summary(1.8),
            "test": _summary(2.1),
            "static_benchmark": {
                "role": (
                    "external_benchmark_not_rollout_loss_or_checkpoint_ranking"
                ),
                "train": _summary(0.9),
                "validation": _summary(0.7),
                "test": _summary(1.0),
            },
        },
    )


def test_bundle_deployability_uses_only_calmar_and_technical_contract():
    manifest = _manifest()

    manifest.require_deployable()
    assert manifest.bundle_version == BUNDLE_VERSION
    assert manifest.technical_convergence is True


def test_previous_ten_factor_bundle_is_not_an_eleven_factor_bundle():
    payload = _manifest().to_dict()
    payload["bundle_version"] = "wbr-policy-bundle-v32-long-history-actual-actions"
    with pytest.raises(ValueError, match="version"):
        BundleManifest.from_dict(payload).require_deployable()


def test_selected10_bundle_cannot_drop_financial_provenance():
    payload = _manifest().to_dict()
    del payload["runtime"]["financial_snapshot"]
    with pytest.raises(ValueError, match="financial snapshot provenance"):
        BundleManifest.from_dict(payload).require_deployable()


@pytest.mark.parametrize("field", ("manifest_sha256", "snapshot_sha256", "financial_identity_sha256"))
def test_selected10_bundle_rejects_invalid_financial_hash(field):
    payload = _manifest().to_dict()
    payload["runtime"]["financial_snapshot"][field] = "not-a-sealed-source"
    with pytest.raises(ValueError, match=field):
        BundleManifest.from_dict(payload).require_deployable()














def test_bundle_rejects_non_calmar_selection_or_missing_full_investment():
    payload = _manifest().to_dict()
    payload["training"]["checkpoint_selection"]["objective"] = "nine_margins"
    changed = BundleManifest.from_dict(payload)
    with pytest.raises(ValueError, match="validation Calmar"):
        changed.require_deployable()

    payload = _manifest().to_dict()
    payload["evaluation"]["test"]["full_investment_contract_satisfied"] = False
    changed = BundleManifest.from_dict(payload)
    with pytest.raises(ValueError, match="full-investment"):
        changed.require_deployable()

@pytest.mark.parametrize("split", ["train", "validation", "test"])
@pytest.mark.parametrize("calmar", [-1.0, 0.0, 1.5])
def test_finite_performance_has_no_qualification_threshold(split, calmar):
    payload = _manifest().to_dict()
    payload["evaluation"][split]["metrics"]["calmar"] = calmar
    payload["evaluation"]["static_benchmark"][split]["metrics"]["calmar"] = 100.0
    BundleManifest.from_dict(payload).require_deployable()

@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf")])
def test_candidate_evaluation_must_remain_finite(invalid):
    payload = _manifest().to_dict()
    payload["evaluation"]["test"]["metrics"]["calmar"] = invalid
    with pytest.raises(ValueError):
        BundleManifest.from_dict(payload).require_deployable()
