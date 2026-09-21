"""Raw-state actor/critic with the standard SB3 continuous Gaussian action head."""

from __future__ import annotations

from functools import partial
from typing import Any, Mapping

from gymnasium import spaces
import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.type_aliases import Schedule

from env.action_schema import ActionSchema
from env.backtest import CRITIC_CONTEXT_SCALAR_FEATURE_NAMES
from ai.rl.raw_features import RAW_PANEL_CONFIG, RAW_PANEL_NETWORK_VERSION, RawPanelFeatures
from ai.rl.device import PPO_DEVICE, require_cuda_device, require_cuda_input
from ai.rl.inference import FrozenActorGraph


TYPED_ACTION_DISTRIBUTION_VERSION = "wbr-action-distribution-v14-sb3-gaussian-clipped-box"


class AsymmetricRawPanelExtractor(nn.Module):
    """Shared raw features; training-only context enters only the value MLP."""

    def __init__(self, encoded_schema: Mapping[str, object], config: Mapping[str, int],
                 value_arch: list[int], activation_fn: type[nn.Module]) -> None:
        super().__init__()
        self.raw_features = RawPanelFeatures(encoded_schema, config)
        self.actor_feature_dim = self.raw_features.dimension
        self.latent_dim_pi = self.raw_features.width
        self.latent_dim_vf = value_arch[-1]
        self.value_net = nn.Sequential(*create_mlp(
            self.latent_dim_pi + len(CRITIC_CONTEXT_SCALAR_FEATURE_NAMES), -1, value_arch, activation_fn))

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        require_cuda_input(self, features)
        public = self.raw_features(features[:, :self.actor_feature_dim])
        critic = self.value_net(th.cat((public, features[:, self.actor_feature_dim:]), dim=1))
        return public, critic

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        require_cuda_input(self, features)
        return self.raw_features(features[:, :self.actor_feature_dim])

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.forward(features)[1]


class TypedActorCriticPolicy(ActorCriticPolicy):
    """Standard SB3 Gaussian PPO with raw features and isolated critic context."""

    action_dist: DiagGaussianDistribution

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        *,
        action_schema: ActionSchema | Mapping[str, object] | None = None,
        encoded_schema: Mapping[str, object],
        raw_panel_config: Mapping[str, int] | None = None,
        action_head_gain: float = 0.01,
        antithetic_exploration: bool = False,
        **kwargs: Any,
    ) -> None:
        device = require_cuda_device()
        if type(antithetic_exploration) is not bool:
            raise TypeError("antithetic_exploration must be bool")
        self.antithetic_exploration = antithetic_exploration
        if not isinstance(action_head_gain, (int, float)) or isinstance(action_head_gain, bool) or not (
            action_head_gain > 0.0 and np.isfinite(action_head_gain)
        ):
            raise ValueError("action_head_gain must be a positive finite number")
        self.action_head_gain = float(action_head_gain)
        schema = (
            ActionSchema.from_dict(action_schema)
            if isinstance(action_schema, Mapping)
            else action_schema or ActionSchema()
        )
        self.typed_action_schema = schema
        if not isinstance(observation_space, spaces.Box) or len(observation_space.shape) != 1:
            raise TypeError("TypedActorCriticPolicy requires a one-dimensional Box observation")
        full_observation_dim = int(observation_space.shape[0])
        self.encoded_schema = dict(encoded_schema)
        self.actor_observation_dim = int(self.encoded_schema["dimension"])
        self.raw_panel_config = dict(RAW_PANEL_CONFIG if raw_panel_config is None else raw_panel_config)
        if full_observation_dim != self.actor_observation_dim + len(CRITIC_CONTEXT_SCALAR_FEATURE_NAMES):
            raise ValueError("raw policy requires compact public coordinates followed by the declared critic context")
        self._validate_action_space(action_space, schema)
        if bool(kwargs.get("use_sde", False)):
            raise ValueError("TypedActorCriticPolicy does not support gSDE")
        if bool(kwargs.get("squash_output", False)):
            raise ValueError(
                "Standard PPO clips Gaussian actions to the declared Box; "
                "squash_output must be false"
            )
        with th.device(device):
            super().__init__(
                observation_space,
                action_space,
                lr_schedule,
                **kwargs,
            )
        if self.ortho_init and self.action_head_gain != 0.01:
            # SB3 hard-codes orthogonal gain 0.01 for the mean head. A larger gain lets the
            # state-dependent latent reach the action from the first update (2026-09-21 probe);
            # the same nn.Parameter objects stay registered in the optimizer.
            self.action_net.apply(partial(self.init_weights, gain=self.action_head_gain))
        self._frozen_actor_graph = None
        self._frozen_actor_key = None

    @staticmethod
    def _validate_action_space(
        action_space: spaces.Space,
        action_schema: ActionSchema,
    ) -> None:
        if not isinstance(action_space, spaces.Box):
            raise TypeError("TypedActorCriticPolicy requires a Box action space")
        if action_space.shape != (action_schema.action_dim,):
            raise ValueError(
                "policy action space and ActionSchema dimensions differ: "
                f"{action_space.shape} != ({action_schema.action_dim},)"
            )
        if action_space.dtype != np.dtype(np.float32):
            raise ValueError("typed action space dtype must be float32")
        if not (
            np.array_equal(action_space.low, np.full(action_schema.action_dim, -1.0))
            and np.array_equal(action_space.high, np.full(action_schema.action_dim, 1.0))
        ):
            raise ValueError("typed action space bounds must be exactly [-1, 1]")

    def _build_mlp_extractor(self) -> None:
        if self.features_dim != int(self.observation_space.shape[0]):
            raise ValueError("raw policy requires an identity-sized feature extractor")
        value_arch = (list(self.net_arch["vf"]) if isinstance(self.net_arch, dict)
                      else list(self.net_arch))
        if not value_arch:
            raise ValueError("raw policy requires a nonempty critic MLP")
        self.mlp_extractor = AsymmetricRawPanelExtractor(
            self.encoded_schema, self.raw_panel_config, value_arch, self.activation_fn,
        )

    def _invalidate_frozen_actor(self) -> None:
        self._frozen_actor_graph = None
        self._frozen_actor_key = None

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """SB3 forward with optional antithetic (mirrored) exploration noise across the batch.

        With ``antithetic_exploration`` the rollout batch of N lock-step environments draws N/2
        Gaussian perturbations and applies each once with sign +1 and once with sign -1. Every
        environment's action is still an exact draw from its own N(mean, std) marginal, so the
        stored log-probabilities and the clipped PPO surrogate are unchanged; only the paired
        structure reduces the variance of the row-mean-centered advantage (antithetic variates).
        """
        if deterministic or not self.antithetic_exploration:
            return super().forward(obs, deterministic=deterministic)
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        mean = distribution.distribution.mean
        std = distribution.distribution.stddev
        half, odd = divmod(mean.shape[0], 2)
        noise = th.randn((half + odd, mean.shape[1]), device=mean.device, dtype=mean.dtype)
        mirrored = th.cat((noise, -noise[:half]), dim=0)
        actions = mean + std * mirrored
        log_prob = distribution.log_prob(actions)
        return actions.reshape((-1, *self.action_space.shape)), values, log_prob

    def train(self, mode: bool = True):
        if mode:
            self._invalidate_frozen_actor()
        return super().train(mode)

    def _frozen_actions(self, public_observation: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        parameters = self.action_net(self.mlp_extractor.raw_features.forward_frozen_tensor(public_observation))
        return parameters, th.isfinite(parameters).all() & (self.log_std.exp() > 0).all()

    @th.no_grad()
    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        if not deterministic:
            return super()._predict(observation, deterministic=False)
        require_cuda_input(self, observation)
        if observation.ndim != 2 or observation.shape[1:] != self.observation_space.shape:
            raise ValueError("PPO inference observation does not match the model layout")
        public = observation[:, :self.actor_observation_dim]
        raw = self.mlp_extractor.raw_features
        raw.validate_references(public)
        bank = raw.prepare_frozen_market()
        key = (tuple(public.shape), public.device, public.dtype,
               raw.frozen_market_token, bank.data_ptr(),
               tuple((id(parameter), parameter.data_ptr(), parameter._version)
                     for parameter in self.parameters()))
        if key != self._frozen_actor_key:
            self._frozen_actor_graph = FrozenActorGraph(self._frozen_actions, public)
            self._frozen_actor_key = key
        actions, valid = self._frozen_actor_graph(public)
        if not bool(valid):
            raise ValueError("non-finite Gaussian mean or nonpositive Gaussian scale")
        return actions.clone()

    def _get_constructor_parameters(self) -> dict[str, Any]:
        parameters = super()._get_constructor_parameters()
        parameters["action_schema"] = self.typed_action_schema.to_dict()
        parameters["encoded_schema"] = dict(self.encoded_schema)
        parameters["raw_panel_config"] = dict(self.raw_panel_config)
        parameters["action_head_gain"] = self.action_head_gain
        parameters["antithetic_exploration"] = self.antithetic_exploration
        return parameters

    def bind_market_store(self, store, normalizer) -> None:
        require_cuda_device(self.device)
        self._invalidate_frozen_actor()
        self.mlp_extractor.raw_features.bind_market_store(store, normalizer)

    def to(self, *args, **kwargs):
        requested = kwargs.get("device")
        if args:
            first = args[0]
            if isinstance(first, (str, th.device, int)):
                requested = first
            elif isinstance(first, th.Tensor):
                requested = first.device
        if requested is not None:
            require_cuda_device(requested)
        result = super().to(*args, **kwargs)
        require_cuda_device(self.device)
        return result

    def cpu(self):
        raise ValueError("PPO training and inference require CUDA; CPU execution is not supported")

    @property
    def market_store(self):
        return self.mlp_extractor.raw_features.market_store

    @property
    def normalizer(self):
        return self.mlp_extractor.raw_features.normalizer

    @classmethod
    def load(
        cls,
        path: str,
        device: th.device | str = PPO_DEVICE,
    ) -> "TypedActorCriticPolicy":
        """Load direct policy checkpoints only on an available CUDA GPU."""

        return super().load(path, device=require_cuda_device(device))


__all__ = [
    "AsymmetricRawPanelExtractor", "RAW_PANEL_CONFIG", "RAW_PANEL_NETWORK_VERSION",
    "TYPED_ACTION_DISTRIBUTION_VERSION", "TypedActorCriticPolicy",
]
