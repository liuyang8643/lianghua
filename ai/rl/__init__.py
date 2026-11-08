"""Stable-Baselines3 PPO training and deterministic inference.

Importing the package has no Torch side effect: spawned rollout workers import
``ai.rl.rollout_worker`` and must stay Torch-free. ``RLPolicy`` is resolved lazily.
"""

from __future__ import annotations

__all__ = ["RLPolicy"]


def __getattr__(name: str):
    if name == "RLPolicy":
        from ai.rl.policy import RLPolicy

        return RLPolicy
    raise AttributeError(f"module 'ai.rl' has no attribute {name!r}")
