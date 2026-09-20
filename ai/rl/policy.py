"""Cold-loadable PPO policy implementing ``env.contracts.Policy``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from stable_baselines3 import PPO

from ai.bundle import BundleManifest, file_sha256, policy_source_sha256
from ai.rl.device import load_cuda_ppo, require_cuda_model
from env.action_schema import ActionSchema
from env.contracts import DayConfig, Observation
from env.encoder import EncodedObservationSchema, ObservationEncoder, RawMarketStore, TrainOnlyNormalizer
from env.backtest import (
    critic_context_dimension,
    environment_schema_manifest,
    neutral_critic_context,
)
from env.observation import ObservationSchema
from env.prefilter import prefilter_n_from_config


def predict_encoded_action(model: PPO, observation: NDArray[np.float32], *,
                           deterministic: bool = True) -> NDArray[np.float32]:
    """Run public encoded state with neutral training-only critic coordinates."""
    require_cuda_model(model)
    public = np.asarray(observation, dtype=np.float32)
    expected_shape = model.observation_space.shape
    if expected_shape is None or len(expected_shape) != 1:
        raise RuntimeError("frozen PPO must expose a flat observation space")
    expected_dim = int(expected_shape[0])
    actor_dim = int(model.policy.actor_observation_dim)
    if public.shape == (expected_dim,):
        model_observation = public
    elif public.shape == (actor_dim,) and actor_dim < expected_dim:
        model_observation = np.concatenate(
            (public, neutral_critic_context(expected_dim - actor_dim))
        )
    else:
        raise ValueError("evaluation observation does not match frozen PPO dimensions: "
                         f"{public.shape} not in {((actor_dim,), (expected_dim,))}")
    action, _ = model.predict(model_observation, deterministic=deterministic)
    return np.asarray(action, dtype=np.float32)


class RLPolicy:
    def __init__(
        self,
        model: PPO,
        action_schema: ActionSchema,
        encoder: ObservationEncoder,
        normalizer: TrainOnlyNormalizer,
        *,
        prefilter_n: int,
        manifest: BundleManifest | None = None,
    ) -> None:
        require_cuda_model(model)
        model_shape = model.observation_space.shape
        if model_shape is None or len(model_shape) != 1:
            raise ValueError("PPO observation space must be one-dimensional")
        model_dimension = int(model_shape[0])
        public_dimension = encoder.output_dimension
        actor_dimension = int(model.policy.actor_observation_dim)
        if model_dimension != public_dimension + critic_context_dimension(encoder):
            raise ValueError("PPO and encoder observation dimensions differ")
        if actor_dimension != public_dimension:
            raise ValueError("PPO actor can access training-only critic context")
        if model.action_space.shape != (action_schema.action_dim,):
            raise ValueError("PPO and action schema dimensions differ")
        if normalizer.encoder_schema != encoder.output_schema.identifier:
            raise ValueError("normalizer and encoder schemas differ")
        if type(prefilter_n) is not int or prefilter_n <= 0:
            raise ValueError("prefilter_n must be a positive int")
        self.model = model
        self.action_schema = action_schema
        self.encoder = encoder
        self.normalizer = normalizer
        self.prefilter_n = prefilter_n
        self.manifest = manifest

    def _predict_action(
        self,
        observation: Observation,
        deterministic: bool = True,
    ) -> NDArray[np.float32]:
        store = RawMarketStore.from_observation(observation, self.encoder)
        self.model.policy.bind_market_store(store, self.normalizer)
        encoded = self.encoder.encode(observation, store=store)
        result = predict_encoded_action(self.model, encoded, deterministic=deterministic)
        if result.shape != (self.action_schema.action_dim,):
            raise RuntimeError("PPO returned an incompatible action shape")
        return result

    def predict(
        self,
        observation: Observation,
        deterministic: bool = True,
    ) -> DayConfig:
        return self.action_schema.decode(
            self._predict_action(observation, deterministic=deterministic)
        )

    @classmethod
    def load(
        cls,
        directory: str | Path,
    ) -> "RLPolicy":
        root = Path(directory)
        manifest = BundleManifest.load(root, verify_files=True)
        manifest.require_deployable()
        if manifest.algorithm != "stable_baselines3.PPO":
            raise ValueError(f"unsupported algorithm: {manifest.algorithm}")
        repo_root = Path(__file__).resolve().parents[2]
        if policy_source_sha256(repo_root) != manifest.source_sha256:
            raise ValueError("policy source semantics differ from the frozen bundle")
        resolved_config = root / manifest.config_file
        if file_sha256(resolved_config) != manifest.config_sha256:
            raise ValueError("strategy config differs from the trained bundle")
        config_payload = json.loads(resolved_config.read_text(encoding="utf-8"))
        if not isinstance(config_payload, dict):
            raise TypeError("strategy config must be a JSON object")
        prefilter_n = prefilter_n_from_config(config_payload)
        observation_schema = ObservationSchema.from_dict(manifest.observation_schema)
        encoder = ObservationEncoder(observation_schema)
        encoded_schema = EncodedObservationSchema.from_dict(manifest.encoded_schema)
        if encoder.output_schema.identifier != encoded_schema.identifier:
            raise ValueError("bundle encoder implementation/schema mismatch")
        action_schema = ActionSchema.from_dict(manifest.action_schema)
        if dict(manifest.environment) != environment_schema_manifest(
            action_schema,
            encoder,
            prefilter_n=prefilter_n,
        ):
            raise ValueError("bundle environment semantics differ from current env")
        normalizer = TrainOnlyNormalizer.load(
            root / manifest.normalizer_file,
            expected_schema=encoded_schema,
        )
        model = load_cuda_ppo(root / manifest.model_file)
        return cls(
            model,
            action_schema,
            encoder,
            normalizer,
            prefilter_n=prefilter_n,
            manifest=manifest,
        )


__all__ = ["RLPolicy", "predict_encoded_action"]
