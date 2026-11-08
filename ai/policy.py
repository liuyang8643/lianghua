"""Small non-learning policies sharing the public env policy contract."""

from __future__ import annotations

from dataclasses import dataclass

from env.contracts import DayConfig, Observation


@dataclass(frozen=True)
class FixedPolicy:
    config: DayConfig

    def predict(
        self,
        observation: Observation,
        deterministic: bool = True,
    ) -> DayConfig:
        del observation, deterministic
        return self.config


__all__ = ["FixedPolicy"]
