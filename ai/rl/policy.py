"""Cold-loadable PPO policy implementing ``env.contracts.Policy``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from stable_baselines3 import PPO

from ai.bundle import BundleManifest, file_sha256, policy_source_sha256
from env.action_schema import ActionSchema
from env.contracts import DayConfig, Observation
from env.encoder import EncodedObservationSchema, ObservationEncoder, TrainOnlyNormalizer
from env.gym_adapter import environment_schema_manifest
from env.observation import ObservationSchema


class RLPolicy:
    def __init__(
        self,
        model: PPO,
        action_schema: ActionSchema,
        encoder: ObservationEncoder,
        normalizer: TrainOnlyNormalizer,
        *,
        manifest: BundleManifest | None = None,
    ) -> None:
        if model.observation_space.shape != (encoder.output_dimension,):
            raise ValueError("PPO and encoder observation dimensions differ")
        if model.action_space.shape != (action_schema.action_dim,):
            raise ValueError("PPO and action schema dimensions differ")
        if normalizer.encoder_schema != encoder.output_schema.identifier:
            raise ValueError("normalizer and encoder schemas differ")
        self.model = model
        self.action_schema = action_schema
        self.encoder = encoder
        self.normalizer = normalizer
        self.manifest = manifest

    def predict_action(
        self,
        observation: Observation,
        deterministic: bool = True,
    ) -> NDArray[np.float32]:
        encoded = self.encoder.encode(observation, normalizer=self.normalizer)
        action, _ = self.model.predict(encoded, deterministic=deterministic)
        result = np.asarray(action, dtype=np.float32)
        if result.shape != (self.action_schema.action_dim,):
            raise RuntimeError("PPO returned an incompatible action shape")
        return result

    def predict(
        self,
        observation: Observation,
        deterministic: bool = True,
    ) -> DayConfig:
        return self.action_schema.decode(
            self.predict_action(observation, deterministic=deterministic)
        )

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        config_path: str | Path | None = None,
    ) -> "RLPolicy":
        root = Path(directory)
        manifest = BundleManifest.load(root, verify_files=True)
        manifest.require_deployable()
        if manifest.algorithm != "stable_baselines3.PPO":
            raise ValueError(f"unsupported algorithm: {manifest.algorithm}")
        repo_root = Path(__file__).resolve().parents[2]
        if policy_source_sha256(repo_root) != manifest.source_sha256:
            raise ValueError("policy source semantics differ from the frozen bundle")
        resolved_config = (
            root / manifest.config_file
            if config_path is None
            else Path(config_path).resolve()
        )
        if file_sha256(resolved_config) != manifest.config_sha256:
            raise ValueError("strategy config differs from the trained bundle")
        observation_schema = ObservationSchema.from_dict(manifest.observation_schema)
        encoder = ObservationEncoder(observation_schema)
        encoded_schema = EncodedObservationSchema.from_dict(manifest.encoded_schema)
        if encoder.output_schema.identifier != encoded_schema.identifier:
            raise ValueError("bundle encoder implementation/schema mismatch")
        action_schema = ActionSchema.from_dict(manifest.action_schema)
        if dict(manifest.environment) != environment_schema_manifest(action_schema):
            raise ValueError("bundle environment semantics differ from current env")
        normalizer = TrainOnlyNormalizer.load(
            root / manifest.normalizer_file,
            expected_schema=encoded_schema,
        )
        model = PPO.load(root / manifest.model_file, device="cpu")
        return cls(
            model,
            action_schema,
            encoder,
            normalizer,
            manifest=manifest,
        )


__all__ = ["RLPolicy"]
