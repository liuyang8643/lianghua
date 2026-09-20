"""The single simulated A-share fee schedule used by planning and fills."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import math
from numba.extending import register_jitable
from typing import Iterable

from env.contracts import Fill


def summarize_fill_costs(fills: Iterable[Fill]) -> tuple[float, float]:
    """Actual traded notional and costs in execution order for both runtimes."""
    gross_notional = 0.0
    total_cost = 0.0
    for fill in fills:
        gross_notional += int(fill.quantity) * float(fill.price)
        total_cost += float(fill.fee)
    return gross_notional, total_cost


@register_jitable(inline='always')
def broker_commission(notional, parameters):
    return max(notional * parameters[0], parameters[1])


@register_jitable(inline='always')
def buy_fee_components(notional, parameters):
    commission = broker_commission(notional, parameters)
    transfer = notional * parameters[3]
    slippage = notional * parameters[4]
    return commission, transfer, slippage, commission + transfer + slippage


@register_jitable(inline='always')
def buy_fee(notional, parameters):
    return buy_fee_components(notional, parameters)[3]


@register_jitable(inline='always')
def buy_total_cost(notional, parameters):
    return notional + buy_fee(notional, parameters)


@register_jitable(inline='always')
def sell_fee(notional, parameters):
    return buy_fee(notional, parameters) + notional * parameters[2]


@register_jitable(inline='always')
def sell_net_proceeds(notional, parameters):
    return notional - sell_fee(notional, parameters)


@register_jitable
def affordable_buy_shares(cash, unit_price, parameters):
    """Exact inversion of the shared fee law, usable by serial numeric kernels."""
    if cash <= 0.0 or unit_price <= 0.0:
        return 0
    variable = 1.0 + parameters[3] + parameters[4]
    notional = min(cash / (variable + parameters[0]), (cash - parameters[1]) / variable)
    upper = int(cash / unit_price)
    quantity = min(upper, max(0, int(notional / unit_price)))
    while quantity > 0 and buy_total_cost(quantity * unit_price, parameters) > cash:
        quantity -= 1
    while quantity < upper and buy_total_cost((quantity + 1) * unit_price, parameters) <= cash:
        quantity += 1
    return quantity


@dataclass(frozen=True)
class FeeSchedule:
    commission_rate: float = 0.0000854
    minimum_commission: float = 0.1
    stamp_tax_rate: float = 0.0005
    transfer_fee_rate: float = 0.00002
    slippage_rate: float = 0.0025

    def __post_init__(self) -> None:
        names = (
            "commission_rate",
            "minimum_commission",
            "stamp_tax_rate",
            "transfer_fee_rate",
            "slippage_rate",
        )
        for name in names:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError("fee schedule values must be finite and non-negative")
            object.__setattr__(self, name, float(value))

    def broker_commission(self, notional: float) -> float:
        return broker_commission(float(notional), self.parameters)

    @cached_property
    def parameters(self) -> tuple[float, float, float, float, float]:
        """Numeric terms for the same fee law inside compiled domain kernels."""
        return (self.commission_rate, self.minimum_commission, self.stamp_tax_rate,
                self.transfer_fee_rate, self.slippage_rate)

    def buy_fee(self, notional: float) -> float:
        return buy_fee(float(notional), self.parameters)

    def buy_fee_components(self, notional: float) -> tuple[float, float, float, float]:
        return buy_fee_components(float(notional), self.parameters)

    def sell_fee(self, notional: float) -> float:
        return sell_fee(float(notional), self.parameters)

    def buy_total_cost(self, notional: float) -> float:
        value = float(notional)
        return buy_total_cost(value, self.parameters)

    def sell_net_proceeds(self, notional: float) -> float:
        value = float(notional)
        return sell_net_proceeds(value, self.parameters)

    def affordable_buy_shares(self, cash: float, unit_price: float) -> int:
        """Invert the piecewise-linear buy cost, preserving exact float boundaries.

        Exchange lot rules are applied by the shared quantity module afterwards.
        The final checks use the authoritative fee calculation, so a rounding
        boundary cannot cause a planner/executor disagreement.
        """
        return affordable_buy_shares(cash, unit_price, self.parameters)


DEFAULT_FEE_SCHEDULE = FeeSchedule()


__all__ = ["DEFAULT_FEE_SCHEDULE", "FeeSchedule", "broker_commission", "buy_fee",
           "buy_fee_components", "buy_total_cost", "sell_fee", "sell_net_proceeds", "affordable_buy_shares", "summarize_fill_costs"]
