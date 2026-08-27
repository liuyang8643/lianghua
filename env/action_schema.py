"""The single mixed-type action codec used by GA, PPO, and live inference."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Literal, Mapping, Sequence

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from env.contracts import DayConfig, RebalanceMode
from factor.registry import PRODUCTION_FACTOR_NAMES, PRODUCTION_FILTER_NAMES


CORE_FACTOR_NAMES = PRODUCTION_FACTOR_NAMES
CORE_FILTER_NAMES = PRODUCTION_FILTER_NAMES
DEFAULT_BUY_N_CHOICES = (20, 25, 30, 35, 40, 45, 50, 100)
DEFAULT_SELL_M_CHOICES = (20, 25, 30, 35, 40, 45, 50, 100)


@dataclass(frozen=True)
class ActionField:
    """One stable coordinate in the flat PPO action vector."""

    index: int
    name: str
    kind: Literal["continuous", "binary", "discrete", "enum"]
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[int | str, ...] = ()


@dataclass(frozen=True)
class ActionSchema:
    """Decode one ``Box(-1, 1, D)`` action into a validated ``DayConfig``.

    The factor and filter vocabularies are immutable model structure.  Changing
    either tuple changes the action layout and therefore requires a new model.
    """

    factor_names: tuple[str, ...] = CORE_FACTOR_NAMES
    filter_names: tuple[str, ...] = CORE_FILTER_NAMES
    buy_n_choices: tuple[int, ...] = DEFAULT_BUY_N_CHOICES
    sell_m_choices: tuple[int, ...] = DEFAULT_SELL_M_CHOICES
    exposure_range: tuple[float, float] = (0.0, 1.0)
    rebalance_band_range: tuple[float, float] = (0.0, 0.15)
    schema_version: str = "day-config-v1"
    _layout: tuple[ActionField, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.factor_names or len(set(self.factor_names)) != len(self.factor_names):
            raise ValueError("factor_names must be non-empty and unique")
        if not self.filter_names or len(set(self.filter_names)) != len(self.filter_names):
            raise ValueError("filter_names must be non-empty and unique")
        self._validate_choices("buy_n_choices", self.buy_n_choices)
        self._validate_choices("sell_m_choices", self.sell_m_choices)
        self._validate_continuous_range("exposure_range", self.exposure_range, upper_closed=True)
        self._validate_continuous_range(
            "rebalance_band_range", self.rebalance_band_range, upper_closed=False
        )

        layout: list[ActionField] = []
        for name in self.factor_names:
            layout.append(ActionField(len(layout), f"factor_weight.{name}", "continuous", 0.0, 1.0))
        for name in self.factor_names:
            layout.append(ActionField(len(layout), f"factor_enabled.{name}", "binary"))
        for name in self.filter_names:
            layout.append(ActionField(len(layout), f"filter_flag.{name}", "binary"))
        layout.extend(
            (
                ActionField(
                    len(layout),
                    "target_exposure",
                    "continuous",
                    self.exposure_range[0],
                    self.exposure_range[1],
                ),
                ActionField(len(layout) + 1, "buy_n", "discrete", choices=self.buy_n_choices),
                ActionField(len(layout) + 2, "sell_m", "discrete", choices=self.sell_m_choices),
                ActionField(len(layout) + 3, "rebalance_now", "binary"),
                ActionField(
                    len(layout) + 4,
                    "rebalance_mode",
                    "enum",
                    choices=tuple(mode.value for mode in RebalanceMode),
                ),
                ActionField(len(layout) + 5, "limit_up_protection", "binary"),
                ActionField(
                    len(layout) + 6,
                    "rebalance_band_pct",
                    "continuous",
                    self.rebalance_band_range[0],
                    self.rebalance_band_range[1],
                ),
            )
        )
        object.__setattr__(self, "_layout", tuple(layout))

    @staticmethod
    def _validate_choices(name: str, choices: Sequence[int]) -> None:
        if not choices or tuple(sorted(set(choices))) != tuple(choices):
            raise ValueError(f"{name} must contain unique ascending values")
        if any(type(value) is not int or value <= 0 for value in choices):
            raise ValueError(f"{name} values must be positive ints")

    @staticmethod
    def _validate_continuous_range(
        name: str,
        value_range: tuple[float, float],
        *,
        upper_closed: bool,
    ) -> None:
        low, high = value_range
        valid_high = high <= 1.0 if upper_closed else high < 1.0
        if not (math.isfinite(low) and math.isfinite(high) and 0.0 <= low < high and valid_high):
            upper = "]" if upper_closed else ")"
            raise ValueError(f"{name} must be an increasing subset of [0, 1{upper}")

    @property
    def layout(self) -> tuple[ActionField, ...]:
        return self._layout

    @property
    def action_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self._layout)

    @property
    def action_dim(self) -> int:
        return len(self._layout)

    @property
    def space_bounds(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        return (
            np.full(self.action_dim, -1.0, dtype=np.float32),
            np.full(self.action_dim, 1.0, dtype=np.float32),
        )

    @property
    def action_space(self) -> spaces.Box:
        low, high = self.space_bounds
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _schema_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "factor_names": list(self.factor_names),
            "filter_names": list(self.filter_names),
            "buy_n_choices": list(self.buy_n_choices),
            "sell_m_choices": list(self.sell_m_choices),
            "exposure_range": list(self.exposure_range),
            "rebalance_band_range": list(self.rebalance_band_range),
            "layout": [
                {
                    "index": item.index,
                    "name": item.name,
                    "kind": item.kind,
                    "minimum": item.minimum,
                    "maximum": item.maximum,
                    "choices": list(item.choices),
                }
                for item in self.layout
            ],
        }

    @property
    def schema_hash(self) -> str:
        encoded = json.dumps(
            self._schema_payload(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, object]:
        payload = self._schema_payload()
        payload["schema_hash"] = self.schema_hash
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ActionSchema":
        schema = cls(
            schema_version=str(payload["schema_version"]),
            factor_names=tuple(str(value) for value in payload["factor_names"]),
            filter_names=tuple(str(value) for value in payload["filter_names"]),
            buy_n_choices=tuple(int(value) for value in payload["buy_n_choices"]),
            sell_m_choices=tuple(int(value) for value in payload["sell_m_choices"]),
            exposure_range=tuple(float(value) for value in payload["exposure_range"]),
            rebalance_band_range=tuple(
                float(value) for value in payload["rebalance_band_range"]
            ),
        )
        if str(payload["schema_hash"]) != schema.schema_hash:
            raise ValueError("action schema hash mismatch")
        if payload.get("layout") != schema._schema_payload()["layout"]:
            raise ValueError("action schema layout mismatch")
        return schema

    def decode(self, action: Sequence[float] | NDArray[np.floating]) -> DayConfig:
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (self.action_dim,):
            raise ValueError(f"action must have shape ({self.action_dim},), got {values.shape}")
        if not np.isfinite(values).all() or np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError("action values must be finite and in [-1, 1]")

        cursor = 0
        raw_weights = self._decode_continuous_array(values[cursor : cursor + len(self.factor_names)])
        cursor += len(self.factor_names)
        enabled_values = values[cursor : cursor + len(self.factor_names)] >= 0.0
        cursor += len(self.factor_names)
        if not enabled_values.any():
            enabled_values[int(np.argmax(raw_weights))] = True
        masked_weights = raw_weights * enabled_values
        weight_sum = float(masked_weights.sum())
        if weight_sum == 0.0:
            masked_weights = enabled_values.astype(np.float64) / float(enabled_values.sum())
        else:
            masked_weights /= weight_sum

        filter_values = values[cursor : cursor + len(self.filter_names)] >= 0.0
        cursor += len(self.filter_names)
        target_exposure = self._decode_continuous(values[cursor], self.exposure_range)
        cursor += 1
        buy_n = int(self._decode_choice(values[cursor], self.buy_n_choices))
        cursor += 1
        decoded_sell_m = int(self._decode_choice(values[cursor], self.sell_m_choices))
        sell_m = max(buy_n, decoded_sell_m)
        cursor += 1
        rebalance_now = bool(values[cursor] >= 0.0)
        cursor += 1
        rebalance_mode = RebalanceMode(self._decode_choice(values[cursor], tuple(RebalanceMode)))
        cursor += 1
        limit_up_protection = bool(values[cursor] >= 0.0)
        cursor += 1
        rebalance_band_pct = self._decode_continuous(values[cursor], self.rebalance_band_range)

        return DayConfig(
            factor_weights=dict(zip(self.factor_names, masked_weights.tolist())),
            factor_enabled=dict(zip(self.factor_names, enabled_values.tolist())),
            filter_flags=dict(zip(self.filter_names, filter_values.tolist())),
            target_exposure=target_exposure,
            buy_n=buy_n,
            sell_m=sell_m,
            rebalance_now=rebalance_now,
            rebalance_mode=rebalance_mode,
            limit_up_protection=limit_up_protection,
            rebalance_band_pct=rebalance_band_pct,
        )

    def encode(self, config: DayConfig) -> NDArray[np.float32]:
        self.validate_day_config(config)
        action: list[float] = []
        action.extend(self._encode_continuous(config.factor_weights[name], (0.0, 1.0)) for name in self.factor_names)
        action.extend(1.0 if config.factor_enabled[name] else -1.0 for name in self.factor_names)
        action.extend(1.0 if config.filter_flags[name] else -1.0 for name in self.filter_names)
        action.append(self._encode_continuous(config.target_exposure, self.exposure_range))
        action.append(self._encode_choice(config.buy_n, self.buy_n_choices))
        action.append(self._encode_choice(config.sell_m, self.sell_m_choices))
        action.append(1.0 if config.rebalance_now else -1.0)
        action.append(self._encode_choice(config.rebalance_mode, tuple(RebalanceMode)))
        action.append(1.0 if config.limit_up_protection else -1.0)
        action.append(self._encode_continuous(config.rebalance_band_pct, self.rebalance_band_range))
        return np.asarray(action, dtype=np.float32)

    def validate_day_config(self, config: DayConfig) -> None:
        if tuple(config.factor_weights) != self.factor_names:
            raise ValueError("DayConfig factor order does not match ActionSchema")
        if tuple(config.factor_enabled) != self.factor_names:
            raise ValueError("DayConfig factor_enabled order does not match ActionSchema")
        if tuple(config.filter_flags) != self.filter_names:
            raise ValueError("DayConfig filter order does not match ActionSchema")
        if config.buy_n not in self.buy_n_choices:
            raise ValueError(f"buy_n {config.buy_n} is not registered in ActionSchema")
        if config.sell_m not in self.sell_m_choices:
            raise ValueError(f"sell_m {config.sell_m} is not registered in ActionSchema")
        exposure_low, exposure_high = self.exposure_range
        if not exposure_low <= config.target_exposure <= exposure_high:
            raise ValueError("target_exposure is outside ActionSchema range")
        band_low, band_high = self.rebalance_band_range
        if not band_low <= config.rebalance_band_pct <= band_high:
            raise ValueError("rebalance_band_pct is outside ActionSchema range")

    def from_static_config(self, payload: Mapping[str, object]) -> DayConfig:
        """Convert a legacy static config (or its outer payload) without prefiltering."""

        nested = payload.get("individual_config")
        config = nested if isinstance(nested, Mapping) else payload
        raw_weights = config.get("weights")
        if not isinstance(raw_weights, Mapping):
            raise TypeError("static config must contain a weights mapping")
        if set(raw_weights) != set(self.factor_names):
            raise ValueError("static factor vocabulary does not match ActionSchema")

        explicit_enabled = config.get("factor_enabled")
        if explicit_enabled is not None and not isinstance(explicit_enabled, Mapping):
            raise TypeError("factor_enabled must be a mapping")
        enabled = {
            name: (
                bool(explicit_enabled[name])
                if isinstance(explicit_enabled, Mapping) and name in explicit_enabled
                else float(raw_weights[name]) > 0.0
            )
            for name in self.factor_names
        }
        if not any(enabled.values()):
            raise ValueError("static config must enable at least one factor")
        positive_weights: dict[str, float] = {}
        for name in self.factor_names:
            value = float(raw_weights[name])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("static factor weights must be finite and non-negative")
            positive_weights[name] = value if enabled[name] else 0.0
        total = sum(positive_weights.values())
        if total == 0.0:
            count = sum(enabled.values())
            weights = {name: (1.0 / count if enabled[name] else 0.0) for name in self.factor_names}
        else:
            weights = {name: positive_weights[name] / total for name in self.factor_names}

        raw_filters = config.get("filter_factors", {})
        if not isinstance(raw_filters, Mapping):
            raise TypeError("filter_factors must be a mapping")
        filters = {name: bool(raw_filters.get(name, False)) for name in self.filter_names}
        reserve = float(config.get("cash_reserve_ratio", 0.0))
        holding_period = int(config.get("holding_period", 1))
        rebalance_mode = (
            RebalanceMode.EQUALIZE if bool(config.get("rebalance", True)) else RebalanceMode.REPLACE_ONLY
        )

        return DayConfig(
            factor_weights=weights,
            factor_enabled=enabled,
            filter_flags=filters,
            target_exposure=1.0 - reserve,
            buy_n=int(config["buy_n"]),
            sell_m=int(config.get("sell_m", config["buy_n"])),
            rebalance_now=bool(config.get("rebalance_now", holding_period == 1)),
            rebalance_mode=rebalance_mode,
            limit_up_protection=bool(config.get("limit_up_protection", False)),
            rebalance_band_pct=float(config.get("rebalance_band_pct", 0.01)),
        )

    def encode_static_config(self, payload: Mapping[str, object]) -> NDArray[np.float32]:
        return self.encode(self.from_static_config(payload))

    def to_static_config(self, config: DayConfig) -> dict[str, object]:
        """Export the legacy parameter vocabulary for parity diagnostics.

        ``rebalance_now`` is a per-day decision and has no general legacy
        ``holding_period`` equivalent.  A true value is exactly representable as
        ``holding_period=1``; the returned explicit flag preserves false values
        for consumers of the new contract.
        """

        self.validate_day_config(config)
        return {
            "weights": dict(config.factor_weights),
            "factor_enabled": dict(config.factor_enabled),
            "filter_factors": dict(config.filter_flags),
            "buy_n": config.buy_n,
            "sell_m": config.sell_m,
            "rebalance": config.rebalance_mode is RebalanceMode.EQUALIZE,
            "rebalance_now": config.rebalance_now,
            "limit_up_protection": config.limit_up_protection,
            "cash_reserve_ratio": 1.0 - config.target_exposure,
            "holding_period": 1,
            "rebalance_band_pct": config.rebalance_band_pct,
        }

    @staticmethod
    def _decode_continuous_array(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return (values + 1.0) / 2.0

    @staticmethod
    def _decode_continuous(value: float, value_range: tuple[float, float]) -> float:
        low, high = value_range
        return low + ((float(value) + 1.0) / 2.0) * (high - low)

    @staticmethod
    def _encode_continuous(value: float, value_range: tuple[float, float]) -> float:
        low, high = value_range
        if not low <= float(value) <= high:
            raise ValueError(f"continuous value {value} is outside [{low}, {high}]")
        return 2.0 * ((float(value) - low) / (high - low)) - 1.0

    @staticmethod
    def _decode_choice(value: float, choices: Sequence[int | str | RebalanceMode]):
        index = min(int(((float(value) + 1.0) / 2.0) * len(choices)), len(choices) - 1)
        return choices[index]

    @staticmethod
    def _encode_choice(value, choices: Sequence[int | str | RebalanceMode]) -> float:
        try:
            index = choices.index(value)
        except ValueError as exc:
            raise ValueError(f"discrete value {value!r} is not registered") from exc
        return -1.0 + 2.0 * (index + 0.5) / len(choices)
