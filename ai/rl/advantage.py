"""Synchronized-environment advantage baseline for standard SB3 PPO.

All rollout environments in ``full`` episode scope advance through identical
decision dates in lockstep. The market shock of a date is therefore shared by
every account chain in the same time-major buffer row and cannot be predicted
by the critic. Subtracting the per-row mean advantage removes that common
component before the standard clipped surrogate is applied, leaving only
relative, action-attributable credit — the paired-path comparison GA obtains by
evaluating every candidate on the same complete period. Returns (critic
targets), clipping, epochs and the optimizer are unchanged.
"""

from __future__ import annotations

import numpy as np
import torch as th

from stable_baselines3.common.buffers import RolloutBuffer


ADVANTAGE_BASELINE_VERSION = "wbr-ppo-advantage-baseline-v1-synchronized-env-row-mean"


class SynchronizedEnvBaselineRolloutBuffer(RolloutBuffer):
    """GAE advantages centered across environments within each buffer row."""

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        super().compute_returns_and_advantage(last_values, dones)
        if self.n_envs < 2:
            raise ValueError("synchronized-environment baseline needs at least two environments")
        self.advantages -= self.advantages.mean(axis=1, keepdims=True)


__all__ = ["ADVANTAGE_BASELINE_VERSION", "SynchronizedEnvBaselineRolloutBuffer"]
