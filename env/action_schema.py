"""The single continuous action codec used by GA, PPO, and live inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
import hashlib
import json
import math
from typing import Literal, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from env.contracts import DayConfig, PolicyHistory, PolicyMemory, decode_unit_action, encode_unit_action
from factor.registry import PRODUCTION_FACTOR_NAMES, PRODUCTION_FILTER_NAMES


CORE_FACTOR_NAMES = PRODUCTION_FACTOR_NAMES
CORE_FILTER_NAMES = PRODUCTION_FILTER_NAMES
FIXED_BUY_N = 50
SERIALIZED_DAY_CONFIG_FIELDS = (
    "weights",
    "factor_enabled",
    "filter_factors",
    "buy_n",
    "turnover_rate",
    "limit_up_protection",
    "rebalance_band_pct",
    "single_buy_pct",
)
STATIC_CONFIG_FIELDS = frozenset(
    field for field in (*SERIALIZED_DAY_CONFIG_FIELDS, "prefilter_n")
    if field != "factor_enabled"
)
STATIC_CONFIG_WRAPPER_FIELDS = frozenset(("ga_profile", "individual_config"))


@dataclass(frozen=True)
class ActionField:
    """One stable coordinate in the flat PPO action vector."""

    index: int
    name: str
    kind: Literal["continuous"]
    minimum: float
    maximum: float
    transform: Literal["linear"] = "linear"


@dataclass(frozen=True)
class ActionSchema:
    """Decode one ``Box(-1, 1, D)`` action into a validated ``DayConfig``.

    The factor and filter vocabularies are immutable model structure.  Changing
    either tuple changes the action layout and therefore requires a new model.
    """

    factor_names: tuple[str, ...] = CORE_FACTOR_NAMES
    filter_names: tuple[str, ...] = CORE_FILTER_NAMES
    fixed_buy_n: int = FIXED_BUY_N
    turnover_maximum: float = 0.2
    fixed_filter_flags: tuple[bool, ...] = (True, True)
    fixed_limit_up_protection: bool = True
    fixed_rebalance_band_pct: float = 0.01
    schema_version: str = "day-config-v19-closed-unit-weights"
    _layout: tuple[ActionField, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Own immutable vocabulary/control sequences before caching identity.
        for name in ("factor_names", "filter_names", "fixed_filter_flags"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.factor_names or len(set(self.factor_names)) != len(self.factor_names):
            raise ValueError("factor_names must be non-empty and unique")
        if not self.filter_names or len(set(self.filter_names)) != len(self.filter_names):
            raise ValueError("filter_names must be non-empty and unique")
        if type(self.fixed_buy_n) is not int or self.fixed_buy_n <= 0:
            raise ValueError("fixed_buy_n must be a positive int")
        if isinstance(self.turnover_maximum, bool) or not math.isfinite(self.turnover_maximum) or not 0 < self.turnover_maximum <= 1:
            raise ValueError("turnover_maximum must be finite and in (0, 1]")
        object.__setattr__(self, "turnover_maximum", float(self.turnover_maximum))
        if len(self.fixed_filter_flags) != len(self.filter_names) or any(
            type(value) is not bool for value in self.fixed_filter_flags
        ):
            raise ValueError("fixed_filter_flags must match filter_names")
        if type(self.fixed_limit_up_protection) is not bool:
            raise TypeError("fixed_limit_up_protection must be bool")
        if not math.isfinite(self.fixed_rebalance_band_pct) or not (
            0.0 <= self.fixed_rebalance_band_pct < 1.0
        ):
            raise ValueError("fixed_rebalance_band_pct must be in [0, 1)")

        layout: list[ActionField] = []
        for name in self.factor_names:
            layout.append(ActionField(len(layout), f"factor_weight.{name}", "continuous", 0.0, 1.0))
        layout.append(ActionField(len(layout), "turnover_rate", "continuous", 0.0, self.turnover_maximum))
        object.__setattr__(self, "_layout", tuple(layout))

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

    def _schema_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "factor_names": list(self.factor_names),
            "filter_names": list(self.filter_names),
            "fixed_buy_n": self.fixed_buy_n,
            "turnover_maximum": self.turnover_maximum,
            "replacement_rule": "count(canonical_box_float32(rate) >= canonical_box_float32(k/buy_n), k=1..buy_n); worst-held-only; outside-full-PIT-top-buy_n",
            "canonical_precision": {
                "coordinates": "field-relative unit IEEE-754 binary32 projected to Box[-1,1]",
                "unit_encode": "float32(float32(2)*float32(value)-float32(1))",
                "unit_decode": "(float64(action)+1)/2",
                "decoded_config": "field_min + unit_decode(unit_encode(unit_decode(action))) * (field_max-field_min)",
                "turnover_boundaries": "actual turnover k/buy_n; equality enters quantity k; actor uses declared field-relative range",
            },
            "fixed_filter_flags": list(self.fixed_filter_flags),
            "fixed_limit_up_protection": self.fixed_limit_up_protection,
            "fixed_rebalance_band_pct": self.fixed_rebalance_band_pct,
            "layout": [
                {
                    "index": item.index,
                    "name": item.name,
                    "kind": item.kind,
                    "minimum": item.minimum,
                    "maximum": item.maximum,
                    "transform": item.transform,
                }
                for item in self.layout
            ],
        }

    @cached_property
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
            fixed_buy_n=int(payload["fixed_buy_n"]),
            turnover_maximum=payload["turnover_maximum"],
            fixed_filter_flags=tuple(bool(value) for value in payload["fixed_filter_flags"]),
            fixed_limit_up_protection=bool(payload["fixed_limit_up_protection"]),
            fixed_rebalance_band_pct=float(payload["fixed_rebalance_band_pct"]),
        )
        if str(payload["schema_hash"]) != schema.schema_hash:
            raise ValueError("action schema hash mismatch")
        if dict(payload) != schema.to_dict():
            raise ValueError("action schema payload mismatch")
        return schema

    def decode(self, action: Sequence[float] | NDArray[np.floating]) -> DayConfig:
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (self.action_dim,):
            raise ValueError(f"action must have shape ({self.action_dim},), got {values.shape}")
        if not np.isfinite(values).all() or np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError("action values must be finite and in [-1, 1]")

        unit_values = decode_unit_action(encode_unit_action(decode_unit_action(values)))
        raw_weights = unit_values[:-1]
        enabled_values = raw_weights != 0.0
        buy_n = self.fixed_buy_n
        turnover_field = self.layout[-1]
        turnover_rate = float(
            turnover_field.minimum
            + unit_values[-1] * (turnover_field.maximum - turnover_field.minimum)
        )
        return DayConfig(
            factor_weights=dict(zip(self.factor_names, raw_weights.tolist())),
            factor_enabled=dict(zip(self.factor_names, enabled_values.tolist())),
            filter_flags=dict(zip(self.filter_names, self.fixed_filter_flags)),
            buy_n=buy_n,
            turnover_rate=turnover_rate,
            limit_up_protection=self.fixed_limit_up_protection,
            rebalance_band_pct=self.fixed_rebalance_band_pct,
            single_buy_pct=1.0 / buy_n,
        )

    def encode(self, config: DayConfig) -> NDArray[np.float32]:
        self.validate_day_config(config)
        turnover_field = self.layout[-1]
        turnover_unit = (
            config.turnover_rate - turnover_field.minimum
        ) / (turnover_field.maximum - turnover_field.minimum)
        unit_values = np.asarray(
            [*(config.factor_weights[name] for name in self.factor_names), turnover_unit],
            dtype=np.float64,
        )
        return encode_unit_action(unit_values)

    def canonicalize_day_config(self, config: DayConfig) -> DayConfig:
        """Put GA/static inputs on exactly the same finite action axis as PPO."""
        return self.decode(self.encode(config))

    def validate_policy_memory(self, memory: PolicyMemory) -> None:
        """Require recorded history before an initialized state enters an actor."""
        if not isinstance(memory, PolicyMemory):
            raise TypeError("policy memory must be PolicyMemory")
        if not memory.initialized:
            return
        history = memory.history
        if history is None:
            raise ValueError("initialized policy memory is missing its actual dated history")
        if history.action_schema_hash != self.schema_hash or history.values.shape[1] != self.action_dim + 2:
            raise ValueError("policy history action schema mismatch")
        actions = history.values[:, :-2]
        canonical = np.ascontiguousarray(actions, dtype=np.float32)
        if not np.array_equal(actions, canonical):
            raise ValueError("every policy history action must use canonical action coordinates")
        if not np.array_equal(history.values[-1, :-2], self.encode(memory.previous_day_config)):
            raise ValueError("policy history must end with the canonical previous DayConfig")

    def advance_policy_memory(
        self,
        previous: PolicyMemory,
        settled: PolicyMemory,
        *,
        decision_date: str,
        history_length: int,
    ) -> PolicyMemory:
        """Append one actual Fill settlement, encoding its DayConfig exactly once."""
        if type(history_length) is not int or history_length <= 0:
            raise ValueError("policy history length must be a positive int")
        self.validate_policy_memory(previous)
        if not isinstance(settled, PolicyMemory) or not settled.initialized:
            raise ValueError("a recorded decision requires a completed config and Fill settlement")
        if settled.history is not None:
            raise ValueError("a completed decision must be appended exactly once")
        action = self.encode(settled.previous_day_config)
        row = np.asarray((
            *action, settled.previous_gross_turnover_ratio, settled.previous_total_cost_ratio,
        ), dtype=np.float64).reshape(1, -1)
        date = np.asarray([decision_date], dtype="datetime64[D]")
        if previous.history is not None:
            if date[0] <= previous.history.decision_dates[-1]:
                raise ValueError("a policy history decision date cannot repeat or go backwards")
            dates = np.concatenate((previous.history.decision_dates, date))[-history_length:]
            values = np.concatenate((previous.history.values, row), axis=0)[-history_length:]
        else:
            dates, values = date, row
        history = PolicyHistory(dates, values, self.schema_hash)
        return PolicyMemory(
            previous_day_config=settled.previous_day_config,
            previous_gross_turnover_ratio=settled.previous_gross_turnover_ratio,
            previous_total_cost_ratio=settled.previous_total_cost_ratio,
            history=history,
        )

    def validate_day_config(self, config: DayConfig) -> None:
        # DayConfig owns immutable mappings; a successful check remains valid
        # for this immutable schema. Retain one certificate, not an object cache.
        if type(config) is DayConfig and getattr(config, "_validated_action_schema_hash", None) == self.schema_hash:
            return
        if tuple(config.factor_weights) != self.factor_names:
            raise ValueError("DayConfig factor order does not match ActionSchema")
        if tuple(config.factor_enabled) != self.factor_names:
            raise ValueError("DayConfig factor_enabled order does not match ActionSchema")
        if tuple(config.filter_flags) != self.filter_names:
            raise ValueError("DayConfig filter order does not match ActionSchema")
        if config.buy_n != self.fixed_buy_n:
            raise ValueError(f"buy_n must equal the fixed ActionSchema value {self.fixed_buy_n}")
        turnover_field = self.layout[-1]
        if not turnover_field.minimum <= config.turnover_rate <= turnover_field.maximum:
            raise ValueError(
                f"turnover_rate must be in [{turnover_field.minimum}, {turnover_field.maximum}]"
            )
        expected_enabled = {
            name: config.factor_weights[name] != 0.0 for name in self.factor_names
        }
        if dict(config.factor_enabled) != expected_enabled:
            raise ValueError("factor_enabled must be derived from non-zero weights")
        if tuple(config.filter_flags.values()) != self.fixed_filter_flags:
            raise ValueError("filter flags differ from fixed ActionSchema controls")
        if config.limit_up_protection != self.fixed_limit_up_protection:
            raise ValueError("limit-up protection differs from fixed ActionSchema control")
        if config.rebalance_band_pct != self.fixed_rebalance_band_pct:
            raise ValueError("rebalance band differs from fixed ActionSchema control")
        if config.single_buy_pct != 1.0 / config.buy_n:
            raise ValueError("single_buy_pct must equal 1 / buy_n")
        if type(config) is DayConfig:
            object.__setattr__(config, "_validated_action_schema_hash", self.schema_hash)

    def from_static_config(self, payload: Mapping[str, object]) -> DayConfig:
        """Convert the explicit static-policy vocabulary into ``DayConfig``."""

        if not isinstance(payload, Mapping):
            raise TypeError("static config must be a mapping")
        if "individual_config" in payload:
            unexpected_wrapper = sorted(set(payload) - STATIC_CONFIG_WRAPPER_FIELDS)
            if unexpected_wrapper:
                raise ValueError(
                    "static config wrapper contains unexpected fields: "
                    + ", ".join(unexpected_wrapper)
                )
            nested = payload["individual_config"]
            if not isinstance(nested, Mapping):
                raise TypeError("individual_config must be a mapping")
            config = nested
        else:
            config = payload

        unexpected = sorted(set(config) - STATIC_CONFIG_FIELDS)
        if unexpected:
            raise ValueError(
                "static config contains unexpected fields: " + ", ".join(unexpected)
            )
        raw_weights = config.get("weights")
        if not isinstance(raw_weights, Mapping):
            raise TypeError("static config must contain a weights mapping")
        if set(raw_weights) != set(self.factor_names):
            raise ValueError("static factor vocabulary does not match ActionSchema")

        positive_weights: dict[str, float] = {}
        for name in self.factor_names:
            value = float(raw_weights[name])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("static factor weights must be finite and non-negative")
            positive_weights[name] = value
        total = sum(positive_weights.values())
        # All-zero weights are a valid tied-score configuration.
        weights = {name: (positive_weights[name] / total if total > 0.0 else 0.0)
                   for name in self.factor_names}
        enabled = {name: weights[name] > 0.0 for name in self.factor_names}

        raw_filters = config.get("filter_factors", {})
        if not isinstance(raw_filters, Mapping):
            raise TypeError("filter_factors must be a mapping")
        if raw_filters and set(raw_filters) != set(self.filter_names):
            raise ValueError("static filter vocabulary does not match ActionSchema")
        filters = {name: bool(raw_filters.get(name, False)) for name in self.filter_names}
        buy_n = int(config["buy_n"])

        day_config = DayConfig(
            factor_weights=weights,
            factor_enabled=enabled,
            filter_flags=filters,
            buy_n=buy_n,
            turnover_rate=float(config["turnover_rate"]),
            limit_up_protection=bool(config.get("limit_up_protection", False)),
            rebalance_band_pct=float(config.get("rebalance_band_pct", 0.01)),
            single_buy_pct=float(config.get("single_buy_pct", 1.0 / buy_n)),
        )
        return self.canonicalize_day_config(day_config)

    def from_serialized_day_config(self, payload: Mapping[str, object]) -> DayConfig:
        """Restore an exact policy snapshot without legacy config defaults."""

        if not isinstance(payload, Mapping):
            raise TypeError("serialized DayConfig must be a mapping")
        expected_fields = set(SERIALIZED_DAY_CONFIG_FIELDS)
        actual_fields = set(payload)
        if actual_fields != expected_fields:
            missing = sorted(expected_fields - actual_fields)
            unexpected = sorted(actual_fields - expected_fields)
            raise ValueError(
                "serialized DayConfig fields must match exactly; "
                f"missing={missing}, unexpected={unexpected}"
            )

        raw_weights = payload["weights"]
        raw_enabled = payload["factor_enabled"]
        raw_filters = payload["filter_factors"]
        if not isinstance(raw_weights, Mapping):
            raise TypeError("serialized DayConfig weights must be a mapping")
        if not isinstance(raw_enabled, Mapping):
            raise TypeError("serialized DayConfig factor_enabled must be a mapping")
        if not isinstance(raw_filters, Mapping):
            raise TypeError("serialized DayConfig filter_factors must be a mapping")
        if set(raw_weights) != set(self.factor_names):
            raise ValueError("serialized factor weight vocabulary does not match ActionSchema")
        if set(raw_enabled) != set(self.factor_names):
            raise ValueError("serialized factor_enabled vocabulary does not match ActionSchema")
        if set(raw_filters) != set(self.filter_names):
            raise ValueError("serialized filter vocabulary does not match ActionSchema")

        config = DayConfig(
            factor_weights={name: raw_weights[name] for name in self.factor_names},
            factor_enabled={name: raw_enabled[name] for name in self.factor_names},
            filter_flags={name: raw_filters[name] for name in self.filter_names},
            buy_n=payload["buy_n"],
            turnover_rate=payload["turnover_rate"],
            limit_up_protection=payload["limit_up_protection"],
            rebalance_band_pct=payload["rebalance_band_pct"],
            single_buy_pct=payload["single_buy_pct"],
        )
        self.validate_day_config(config)
        return config

    def to_static_config(self, config: DayConfig) -> dict[str, object]:
        """Export only parameters that the daily policy is allowed to control."""

        self.validate_day_config(config)
        return {
            "weights": dict(config.factor_weights),
            "factor_enabled": dict(config.factor_enabled),
            "filter_factors": dict(config.filter_flags),
            "buy_n": config.buy_n,
            "turnover_rate": config.turnover_rate,
            "limit_up_protection": config.limit_up_protection,
            "rebalance_band_pct": config.rebalance_band_pct,
            "single_buy_pct": config.single_buy_pct,
        }

