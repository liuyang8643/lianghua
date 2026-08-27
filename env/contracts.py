"""Stable DTOs shared by the environment, policies, and executors.

This module intentionally contains no strategy implementation.  The contracts
are small, serialisable values so ``ai`` and ``trade`` can depend on ``env``
without either side reaching into the other's internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray


class RebalanceMode(str, Enum):
    """How an active rebalance changes existing positions."""

    EQUALIZE = "equalize"
    REPLACE_ONLY = "replace_only"


@dataclass(frozen=True)
class DayConfig:
    """Strongly typed strategy freedom selected for one decision day.

    ``rebalance_now`` controls whether the strategy actively trades today.
    ``rebalance_mode`` is independent: it selects equalise/replenish versus
    replacement-only behaviour if a rebalance does occur.
    """

    factor_weights: Mapping[str, float]
    factor_enabled: Mapping[str, bool]
    filter_flags: Mapping[str, bool]
    target_exposure: float
    buy_n: int
    sell_m: int
    rebalance_now: bool
    rebalance_mode: RebalanceMode
    limit_up_protection: bool
    rebalance_band_pct: float

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
        if not any(enabled.values()):
            raise ValueError("at least one factor must be enabled")
        for name, weight in weights.items():
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise TypeError(f"factor weight {name!r} must be numeric")
            if not math.isfinite(float(weight)) or not 0.0 <= float(weight) <= 1.0:
                raise ValueError(f"factor weight {name!r} must be finite and in [0, 1]")
            if not enabled[name] and float(weight) != 0.0:
                raise ValueError(f"disabled factor {name!r} must have zero weight")
        enabled_weight = sum(float(weights[name]) for name, is_enabled in enabled.items() if is_enabled)
        if not math.isclose(enabled_weight, 1.0, rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError("enabled factor weights must sum to 1")
        if isinstance(self.target_exposure, bool) or not isinstance(self.target_exposure, (int, float)):
            raise TypeError("target_exposure must be numeric")
        if not math.isfinite(float(self.target_exposure)) or not 0.0 <= float(self.target_exposure) <= 1.0:
            raise ValueError("target_exposure must be finite and in [0, 1]")
        if type(self.buy_n) is not int or self.buy_n <= 0:
            raise ValueError("buy_n must be a positive int")
        if type(self.sell_m) is not int or self.sell_m < self.buy_n:
            raise ValueError("sell_m must be an int greater than or equal to buy_n")
        if type(self.rebalance_now) is not bool:
            raise TypeError("rebalance_now must be bool")
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

        mode = self.rebalance_mode
        if not isinstance(mode, RebalanceMode):
            try:
                mode = RebalanceMode(mode)
            except ValueError as exc:
                raise ValueError(f"unknown rebalance_mode: {self.rebalance_mode!r}") from exc

        # Copy caller-owned mappings so later mutation cannot change a decision.
        object.__setattr__(self, "factor_weights", MappingProxyType(weights))
        object.__setattr__(self, "factor_enabled", MappingProxyType(enabled))
        object.__setattr__(self, "filter_flags", MappingProxyType(filters))
        object.__setattr__(self, "target_exposure", float(self.target_exposure))
        object.__setattr__(self, "rebalance_mode", mode)
        object.__setattr__(self, "rebalance_band_pct", float(self.rebalance_band_pct))


@dataclass(frozen=True)
class Observation:
    """Causal model input available immediately before the T-open decision."""

    stock_panel: NDArray[np.float32]
    market_panel: NDArray[np.float32]
    position_panel: NDArray[np.float32]
    portfolio: NDArray[np.float32]
    feature_mask: NDArray[np.bool_]
    stock_mask: NDArray[np.bool_]
    time_mask: NDArray[np.bool_]
    schema_version: str
    decision_date: str = ""

    def __post_init__(self) -> None:
        stock_panel = np.asarray(self.stock_panel, dtype=np.float32)
        market_panel = np.asarray(self.market_panel, dtype=np.float32)
        position_panel = np.asarray(self.position_panel, dtype=np.float32)
        portfolio = np.asarray(self.portfolio, dtype=np.float32)
        feature_mask = np.asarray(self.feature_mask, dtype=np.bool_)
        stock_mask = np.asarray(self.stock_mask, dtype=np.bool_)
        time_mask = np.asarray(self.time_mask, dtype=np.bool_)
        if stock_panel.ndim != 3:
            raise ValueError("stock_panel must have shape [L, N, F]")
        if market_panel.ndim != 2 or market_panel.shape[0] != stock_panel.shape[0]:
            raise ValueError("market_panel must have shape [L, M]")
        if position_panel.ndim != 2 or position_panel.shape[0] != stock_panel.shape[1]:
            raise ValueError("position_panel must have shape [N, H]")
        if portfolio.ndim != 1:
            raise ValueError("portfolio must have shape [P]")
        if feature_mask.shape != stock_panel.shape:
            raise ValueError("feature_mask must have shape [L, N, F]")
        if stock_mask.shape != stock_panel.shape[:2]:
            raise ValueError("stock_mask must have shape [L, N]")
        if time_mask.shape != stock_panel.shape[:1]:
            raise ValueError("time_mask must have shape [L]")
        if not self.schema_version:
            raise ValueError("schema_version must not be empty")
        for name, values in (
            ("stock_panel", stock_panel),
            ("market_panel", market_panel),
            ("position_panel", position_panel),
            ("portfolio", portfolio),
        ):
            if not np.isfinite(values).all():
                raise ValueError(f"{name} must be finite; represent missing values with masks")
        object.__setattr__(self, "stock_panel", np.ascontiguousarray(stock_panel))
        object.__setattr__(self, "market_panel", np.ascontiguousarray(market_panel))
        object.__setattr__(self, "position_panel", np.ascontiguousarray(position_panel))
        object.__setattr__(self, "portfolio", np.ascontiguousarray(portfolio))
        object.__setattr__(self, "feature_mask", np.ascontiguousarray(feature_mask))
        object.__setattr__(self, "stock_mask", np.ascontiguousarray(stock_mask))
        object.__setattr__(self, "time_mask", np.ascontiguousarray(time_mask))


@dataclass(frozen=True)
class AccountState:
    cash: float
    positions: Mapping[str, int] = field(default_factory=dict)
    sellable_positions: Mapping[str, int] = field(default_factory=dict)
    average_costs: Mapping[str, float] = field(default_factory=dict)
    last_prices: Mapping[str, float] = field(default_factory=dict)
    nav: float = 0.0
    peak_nav: float = 0.0


@dataclass(frozen=True)
class OrderPlan:
    decision_date: str
    sell_orders: tuple[tuple[str, int], ...] = ()
    buy_orders: Mapping[str, int] = field(default_factory=dict)
    day_config: DayConfig | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
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
