"""One CUDA-only execution and loading contract for every PPO model."""
from __future__ import annotations

from pathlib import Path

import torch as th
from stable_baselines3 import PPO


PPO_DEVICE = "cuda"


def require_cuda_device(device: str | th.device = PPO_DEVICE) -> th.device:
    if str(device) == "auto":
        raise ValueError("PPO requires an explicit CUDA device; auto is not supported")
    resolved = th.device(device)
    if resolved.type != "cuda":
        raise ValueError("PPO training and inference require CUDA; CPU execution is not supported")
    if not th.cuda.is_available():
        raise RuntimeError("PPO requires an available CUDA GPU; CPU fallback is disabled")
    if resolved.index is not None and resolved.index >= th.cuda.device_count():
        raise ValueError(f"PPO CUDA device does not exist: {resolved}")
    return resolved


def require_cuda_model(model: PPO) -> None:
    learner = require_cuda_device(model.device)
    policy = require_cuda_device(model.policy.device)
    if learner.index is not None and learner.index != policy.index:
        raise ValueError("PPO learner and policy must use the same CUDA device")


def require_cuda_input(module: th.nn.Module, values: th.Tensor) -> None:
    device = require_cuda_device(next(module.parameters()).device)
    if values.device != device:
        raise ValueError(f"PPO model inputs must be on {device}; received {values.device}")


def load_cuda_ppo(path: str | Path, *, device: str | th.device = PPO_DEVICE, **kwargs) -> PPO:
    resolved = require_cuda_device(device)
    model = PPO.load(path, device=resolved, **kwargs)
    require_cuda_model(model)
    return model


__all__ = ["PPO_DEVICE", "require_cuda_device", "require_cuda_model", "require_cuda_input", "load_cuda_ppo"]
