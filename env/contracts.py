"""Stable DTOs shared by the environment, policies, and executors.

This module intentionally contains no strategy implementation.  The contracts
are small, serialisable values so ``ai`` and ``trade`` can depend on ``env``
without either side reaching into the other's internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import math
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray


def encode_unit_action(values):
    """Project unit float32 values to the same Box coordinates emitted by PPO."""
    unit = np.asarray(values, dtype=np.float32)
    return np.asarray(np.float32(2.0) * unit - np.float32(1.0), dtype=np.float32)


def decode_unit_action(coordinates):
    """Decode canonical Box coordinates without another float32 arithmetic step."""
    return (np.asarray(coordinates, dtype=np.float64) + 1.0) / 2.0


@lru_cache(maxsize=32)
def _replacement_thresholds(buy_n: int) -> NDArray[np.float32]:
    # These are physical integer-quantity boundaries, not policy categories.
    coordinates = encode_unit_action(np.arange(1, buy_n + 1, dtype=np.float64) / buy_n)
    return np.frombuffer(coordinates.tobytes(), dtype=np.float32)


def replacement_count(buy_n: int, turnover_rate: float) -> int:
    """Count crossed k/buy_n boundaries in the shared canonical float32 codec.

    A value equal to an encoded boundary belongs to the upper quantity bucket.
    Inputs first share the unit-float32 projection used by GA and PPO. Distinct
    canonical coordinates are never absorbed by an epsilon or decimal rounding.
    """
    return int(np.searchsorted(_replacement_thresholds(buy_n),
                               encode_unit_action(turnover_rate), side="right"))


@dataclass(frozen=True)
class DayConfig:
    """Strongly typed strategy freedom selected for one decision day.

    The planner always rebalances towards a fully invested portfolio.  The
    learned fields control selection and concentration, never total exposure.
    """

    factor_weights: Mapping[str, float]
    factor_enabled: Mapping[str, bool]
    filter_flags: Mapping[str, bool]
    buy_n: int
    turnover_rate: float
    limit_up_protection: bool
    rebalance_band_pct: float
    single_buy_pct: float

    def __post_init__(self) -> None:
        weights = dict(self.factor_weights)
        enabled = dict(self.factor_enabled)
        filters = dict(self.filter_flags)
        if not weights or set(weights) != set(enabled):
            raise ValueError("factor_weights and factor_enabled must have identical non-empty keys")
        if any(type(value) is not bool for value in enabled.values()):
            raise TypeError("factor_enabled values must be bool")
        if any(type(value) is not bool for value in filters.values()):
            raise TypeError("filter_flags values must be bool")
        for name, weight in weights.items():
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise TypeError(f"factor weight {name!r} must be numeric")
            if not math.isfinite(float(weight)) or not 0.0 <= float(weight) <= 1.0:
                raise ValueError(f"factor weight {name!r} must be finite and in [0, 1]")
            if not enabled[name] and float(weight) != 0.0:
                raise ValueError(f"disabled factor {name!r} must have zero weight")
            if enabled[name] != (float(weight) != 0.0):
                raise ValueError(f"factor_enabled for {name!r} must equal weight != 0")
        if type(self.buy_n) is not int or self.buy_n <= 0:
            raise ValueError("buy_n must be a positive int")
        if isinstance(self.turnover_rate, bool) or not isinstance(self.turnover_rate, (int, float)):
            raise TypeError("turnover_rate must be numeric")
        if not math.isfinite(self.turnover_rate) or not 0.0 <= self.turnover_rate <= 1.0:
            raise ValueError("turnover_rate must be finite and in [0, 1]")
        if type(self.limit_up_protection) is not bool:
            raise TypeError("limit_up_protection must be bool")
        if isinstance(self.rebalance_band_pct, bool) or not isinstance(
            self.rebalance_band_pct, (int, float)
        ):
            raise TypeError("rebalance_band_pct must be numeric")
        if not math.isfinite(float(self.rebalance_band_pct)) or not 0.0 <= float(
            self.rebalance_band_pct
        ) < 1.0:
            raise ValueError("rebalance_band_pct must be finite and in [0, 1)")
        if isinstance(self.single_buy_pct, bool) or not isinstance(
            self.single_buy_pct, (int, float)
        ):
            raise TypeError("single_buy_pct must be numeric")
        minimum_single_buy_pct = 1.0 / self.buy_n
        if not math.isfinite(float(self.single_buy_pct)) or not minimum_single_buy_pct <= float(
            self.single_buy_pct
        ) <= 1.0:
            raise ValueError(
                "single_buy_pct must be finite and in [1 / buy_n, 1] "
                "so aggregate buy capacity can remain fully invested"
            )

        # Copy caller-owned mappings so later mutation cannot change a decision.
        object.__setattr__(self, "factor_weights", MappingProxyType(weights))
        object.__setattr__(self, "factor_enabled", MappingProxyType(enabled))
        object.__setattr__(self, "filter_flags", MappingProxyType(filters))
        object.__setattr__(self, "rebalance_band_pct", float(self.rebalance_band_pct))
        object.__setattr__(self, "single_buy_pct", float(self.single_buy_pct))

    @property
    def replacement_limit(self) -> int:
        """Maximum worst-ranked holdings examined under canonical rate precision."""
        return replacement_count(self.buy_n, self.turnover_rate)


@dataclass(frozen=True, eq=False)
class PolicyHistory:
    """Dated actual decisions: canonical actions followed by turnover and cost.

    Rows exist only for completed decisions from this account chain. Their
    immutable buffers may be shared with previously emitted observations.
    """

    decision_dates: NDArray[np.datetime64]
    values: NDArray[np.float64]
    action_schema_hash: str

    def __post_init__(self) -> None:
        dates = np.asarray(self.decision_dates, dtype="datetime64[D]")
        values = np.asarray(self.values, dtype=np.float64)
        if dates.ndim != 1 or not dates.size or np.isnat(dates).any():
            raise ValueError("policy history requires non-empty valid decision dates")
        if np.any(dates[1:] <= dates[:-1]):
            raise ValueError("policy history dates must be strictly increasing")
        if values.ndim != 2 or values.shape[0] != len(dates) or values.shape[1] < 3:
            raise ValueError("policy history rows must contain actions, turnover and cost")
        if not np.isfinite(values).all():
            raise ValueError("policy history must be finite")
        if np.any(np.abs(values[:, :-2]) > 1.0) or np.any(values[:, -2:] < 0.0):
            raise ValueError("policy history actions or actual cost ratios are invalid")
        if len(self.action_schema_hash) != 64 or any(
            char not in "0123456789abcdef" for char in self.action_schema_hash
        ):
            raise ValueError("policy history must bind an action schema SHA-256")
        object.__setattr__(self, "decision_dates", np.frombuffer(dates.tobytes(), dtype="datetime64[D]"))
        object.__setattr__(self, "values", np.frombuffer(values.tobytes(), dtype=np.float64).reshape(values.shape))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PolicyHistory):
            return NotImplemented
        return self.action_schema_hash == other.action_schema_hash and np.array_equal(
            self.decision_dates, other.decision_dates,
        ) and np.array_equal(self.values, other.values)

    def __reduce__(self):
        # Re-enter validation and reconstruct immutable buffers on spawn/replay.
        return PolicyHistory, (self.decision_dates, self.values, self.action_schema_hash)


@dataclass(frozen=True)
class PolicyMemory:
    """Causal one-step decision memory available before the next T-open.

    The previous config is the canonical decoded :class:`DayConfig`, while
    turnover and cost ratios come from actual fills divided by pre-trade NAV.
    An empty config marks an explicit cold start.
    """

    previous_day_config: DayConfig | None = None
    previous_gross_turnover_ratio: float = 0.0
    previous_total_cost_ratio: float = 0.0
    history: PolicyHistory | None = None

    def __post_init__(self) -> None:
        if self.previous_day_config is not None and not isinstance(
            self.previous_day_config, DayConfig
        ):
            raise TypeError("previous_day_config must be DayConfig or None")
        for name, value in (
            ("previous_gross_turnover_ratio", self.previous_gross_turnover_ratio),
            ("previous_total_cost_ratio", self.previous_total_cost_ratio),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, float(value))
        if self.previous_day_config is None and (
            self.previous_gross_turnover_ratio != 0.0
            or self.previous_total_cost_ratio != 0.0
            or self.history is not None
        ):
            raise ValueError("cold policy memory must be an atomic zero record")
        if self.history is not None:
            if not isinstance(self.history, PolicyHistory):
                raise TypeError("policy history must be PolicyHistory or None")
            if tuple(self.history.values[-1, -2:]) != (
                self.previous_gross_turnover_ratio, self.previous_total_cost_ratio,
            ):
                raise ValueError("policy history must end with the actual latest turnover and cost")

    @property
    def initialized(self) -> bool:
        return self.previous_day_config is not None


@dataclass(frozen=True)
class Observation:
    """Causal model input available immediately before the T-open decision."""

    stock_panel: NDArray[np.float32]
    position_panel: NDArray[np.float32]
    portfolio: NDArray[np.float32]
    policy_history: NDArray[np.float32]
    time_mask: NDArray[np.bool_]
    pit_universe_mask: NDArray[np.bool_]
    schema_version: str
    decision_date: str = ""

    def __post_init__(self) -> None:
        stock_panel = np.asarray(self.stock_panel, dtype=np.float32)
        position_panel = np.asarray(self.position_panel, dtype=np.float32)
        portfolio = np.asarray(self.portfolio, dtype=np.float32)
        history = np.asarray(self.policy_history, dtype=np.float32)
        time_mask = np.asarray(self.time_mask, dtype=np.bool_)
        pit_universe_mask = np.asarray(self.pit_universe_mask, dtype=np.bool_)
        if stock_panel.ndim != 3:
            raise ValueError("stock_panel must have shape [L, N, F]")
        if position_panel.ndim != 2 or position_panel.shape[0] != stock_panel.shape[1]:
            raise ValueError("position_panel must have shape [N, H]")
        if portfolio.ndim != 1:
            raise ValueError("portfolio must have shape [P]")
        if history.ndim != 2 or history.shape[0] != stock_panel.shape[0] or history.shape[1] < 4:
            raise ValueError("policy_history must have shape [L, valid + actions + turnover + cost]")
        if np.any((history[:, 0] != 0.0) & (history[:, 0] != 1.0)):
            raise ValueError("policy history validity must be binary")
        if np.any(history[history[:, 0] == 0.0] != 0.0):
            raise ValueError("unrecorded policy history rows must be zero padded")
        if time_mask.shape != stock_panel.shape[:1]:
            raise ValueError("time_mask must have shape [L]")
        if pit_universe_mask.shape != stock_panel.shape[:2]:
            raise ValueError("pit_universe_mask must have shape [L, N]")
        if np.any(pit_universe_mask[~time_mask]):
            raise ValueError("padded time rows cannot contain PIT universe members")
        if not self.schema_version:
            raise ValueError("schema_version must not be empty")
        for name, values in (
            ("stock_panel", stock_panel),
            ("position_panel", position_panel),
            ("portfolio", portfolio),
            ("policy_history", history),
        ):
            if not np.isfinite(values).all():
                raise ValueError(f"{name} must be finite; raw unavailable fields use -1")
        object.__setattr__(self, "stock_panel", np.ascontiguousarray(stock_panel))
        object.__setattr__(self, "position_panel", np.ascontiguousarray(position_panel))
        object.__setattr__(self, "portfolio", np.ascontiguousarray(portfolio))
        object.__setattr__(self, "policy_history", np.ascontiguousarray(history))
        object.__setattr__(self, "time_mask", np.ascontiguousarray(time_mask))
        object.__setattr__(
            self,
            "pit_universe_mask",
            np.ascontiguousarray(pit_universe_mask),
        )


@dataclass(frozen=True)
class AccountState:
    cash: float
    positions: Mapping[str, int] = field(default_factory=dict)
    sellable_positions: Mapping[str, int] = field(default_factory=dict)
    average_costs: Mapping[str, float] = field(default_factory=dict)
    last_prices: Mapping[str, float] = field(default_factory=dict)
    mark_provenance: Mapping[str, str] = field(default_factory=dict)
    nav: float = 0.0
    peak_nav: float = 0.0
    max_drawdown: float = 0.0


@dataclass(frozen=True)
class OrderPlan:
    decision_date: str
    sell_orders: tuple[tuple[str, int], ...] = ()
    buy_orders: Mapping[str, int] = field(default_factory=dict)
    day_config: DayConfig | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.decision_date, str) or not self.decision_date:
            raise ValueError("decision_date must be a non-empty string")
        sells = tuple(self.sell_orders)
        sell_codes: set[str] = set()
        for code, quantity in sells:
            if not isinstance(code, str) or not code:
                raise ValueError("sell order codes must be non-empty strings")
            if code in sell_codes:
                raise ValueError("sell order codes must be unique")
            if type(quantity) is not int or quantity <= 0:
                raise ValueError("sell order quantities must be positive ints")
            sell_codes.add(code)
        buys = dict(self.buy_orders)
        for code, quantity in buys.items():
            if not isinstance(code, str) or not code:
                raise ValueError("buy order codes must be non-empty strings")
            if type(quantity) is not int or quantity <= 0:
                raise ValueError("buy order quantities must be positive ints")
        if self.day_config is not None and not isinstance(self.day_config, DayConfig):
            raise TypeError("day_config must be DayConfig or None")
        object.__setattr__(self, "sell_orders", sells)
        object.__setattr__(self, "buy_orders", MappingProxyType(buys))
        object.__setattr__(
            self,
            "diagnostics",
            MappingProxyType(dict(self.diagnostics)),
        )


@dataclass(frozen=True, slots=True)
class Fill:
    code: str
    side: Literal["buy", "sell"]
    quantity: int
    price: float
    fee: float = 0.0
    timestamp: str = ""


@dataclass(frozen=True)
class StepResult:
    account_state: AccountState
    reward: float
    portfolio_return: float
    policy_memory: PolicyMemory = field(default_factory=PolicyMemory)
    fills: tuple[Fill, ...] = ()
    terminated: bool = False
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Policy(Protocol):
    def predict(
        self,
        observation: Observation,
        deterministic: bool = True,
    ) -> DayConfig: ...


@runtime_checkable
class ExecutionPort(Protocol):
    def execute(self, order_plan: OrderPlan) -> Sequence[Fill]: ...
