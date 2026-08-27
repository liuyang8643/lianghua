"""The single simulated A-share fee schedule used by planning and fills."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FeeSchedule:
    commission_rate: float = 0.0000854
    minimum_commission: float = 0.1
    stamp_tax_rate: float = 0.0005
    transfer_fee_rate: float = 0.00002
    slippage_rate: float = 0.001

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
        return max(float(notional) * self.commission_rate, self.minimum_commission)

    def buy_fee(self, notional: float) -> float:
        value = float(notional)
        return (
            self.broker_commission(value)
            + value * self.transfer_fee_rate
            + value * self.slippage_rate
        )

    def sell_fee(self, notional: float) -> float:
        value = float(notional)
        return self.buy_fee(value) + value * self.stamp_tax_rate

    def buy_total_cost(self, notional: float) -> float:
        value = float(notional)
        return value + self.buy_fee(value)

    def sell_net_proceeds(self, notional: float) -> float:
        value = float(notional)
        return value - self.sell_fee(value)


DEFAULT_FEE_SCHEDULE = FeeSchedule()


__all__ = ["DEFAULT_FEE_SCHEDULE", "FeeSchedule"]
