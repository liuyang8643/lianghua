from pathlib import Path

import pytest

from ai.bundle import (
    BUNDLE_VERSION,
    DEPLOYMENT_GATE_NAMES,
    BundleManifest,
    file_sha256,
    source_tree_sha256,
)
from env.action_schema import ActionSchema


def test_action_schema_roundtrip_is_hash_and_layout_strict():
    schema = ActionSchema()
    payload = schema.to_dict()

    assert ActionSchema.from_dict(payload) == schema
    payload["layout"][0]["name"] = "changed"
    with pytest.raises(ValueError, match="action schema"):
        ActionSchema.from_dict(payload)


def test_bundle_manifest_verifies_model_and_normalizer_hashes(tmp_path: Path):
    model = tmp_path / "model.zip"
    normalizer = tmp_path / "normalizer.json"
    config = tmp_path / "strategy_config.json"
    model.write_bytes(b"model")
    normalizer.write_text("{}", encoding="utf-8")
    config.write_text("{}", encoding="utf-8")
    digest = "0" * 64
    manifest = BundleManifest(
        created_at="2026-08-27T00:00:00+00:00",
        algorithm="stable_baselines3.PPO",
        model_file=model.name,
        model_sha256=file_sha256(model),
        normalizer_file=normalizer.name,
        normalizer_sha256=file_sha256(normalizer),
        config_file=config.name,
        config_sha256=file_sha256(config),
        source_sha256=digest,
        runtime={"schema": "runtime"},
        factors={"schema": "factor"},
        observation_schema={"schema": "observation"},
        encoded_schema={"schema": "encoded"},
        action_schema={"schema": "action"},
        environment={"accounting_schema": "test"},
        training={
            "timesteps": 1,
            "convergence": {
                "technical_convergence": True,
                **{name: True for name in DEPLOYMENT_GATE_NAMES},
            },
        },
        evaluation={"finite": True},
    )
    manifest.save(tmp_path)

    loaded = BundleManifest.load(tmp_path)
    assert loaded.model_sha256 == manifest.model_sha256
    with pytest.raises(ValueError, match="missing sealed evaluation"):
        loaded.require_deployable()

    deployable = loaded.to_dict()
    deployable["evaluation"] = {"train": {}, "validation": {}, "test": {}}
    for value in (None, False):
        convergence = deployable["training"]["convergence"]
        if value is None:
            convergence.pop("trained_checkpoint_selected")
        else:
            convergence["trained_checkpoint_selected"] = value
        incomplete = BundleManifest.from_dict(deployable)
        with pytest.raises(
            ValueError,
            match="trained_checkpoint_selected",
        ):
            incomplete.require_deployable()
        convergence["trained_checkpoint_selected"] = True

    with pytest.raises(ValueError, match="runtime prefix lineage"):
        BundleManifest.from_dict(deployable).require_deployable()

    config.write_text('{"tampered": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="strategy_config.json"):
        BundleManifest.load(tmp_path)
    config.write_text("{}", encoding="utf-8")

    normalizer.write_text('{"tampered": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="normalizer.json"):
        BundleManifest.load(tmp_path)
    normalizer.write_text("{}", encoding="utf-8")

    model.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="model.zip"):
        BundleManifest.load(tmp_path)


def test_previous_bundle_schema_is_rejected_explicitly():
    with pytest.raises(ValueError, match="unsupported bundle version"):
        BundleManifest.from_dict({"bundle_version": "wbr-policy-bundle-v4"})

    assert BUNDLE_VERSION == "wbr-policy-bundle-v5"


def test_source_tree_identity_is_independent_of_checkout_line_endings(
    tmp_path: Path,
):
    lf_root = tmp_path / "lf"
    crlf_root = tmp_path / "crlf"
    (lf_root / "pkg").mkdir(parents=True)
    (crlf_root / "pkg").mkdir(parents=True)
    (lf_root / "pkg" / "module.py").write_bytes(b"first\nsecond\n")
    (crlf_root / "pkg" / "module.py").write_bytes(b"first\r\nsecond\r\n")

    assert source_tree_sha256(
        lf_root,
        [lf_root / "pkg" / "module.py"],
    ) == source_tree_sha256(
        crlf_root,
        [crlf_root / "pkg" / "module.py"],
    )


def test_bundle_deployment_requires_explicit_convergence(tmp_path: Path):
    model = tmp_path / "model.zip"
    normalizer = tmp_path / "normalizer.json"
    config = tmp_path / "strategy_config.json"
    model.write_bytes(b"model")
    normalizer.write_text("{}", encoding="utf-8")
    config.write_text("{}", encoding="utf-8")
    digest = "0" * 64
    manifest = BundleManifest(
        created_at="2026-08-27T00:00:00+00:00",
        algorithm="stable_baselines3.PPO",
        model_file=model.name,
        model_sha256=file_sha256(model),
        normalizer_file=normalizer.name,
        normalizer_sha256=file_sha256(normalizer),
        config_file=config.name,
        config_sha256=file_sha256(config),
        source_sha256=digest,
        runtime={},
        factors={},
        observation_schema={},
        encoded_schema={},
        action_schema={},
        environment={},
        training={"convergence": {"technical_convergence": False}},
        evaluation={},
    )
    manifest.save(tmp_path)

    loaded = BundleManifest.load(tmp_path)
    assert not loaded.technical_convergence
    with pytest.raises(ValueError, match="diagnostic-only"):
        loaded.require_deployable()
