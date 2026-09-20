"""Broker execution port that preserves canonical ``OrderPlan`` quantities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import time
from typing import Callable, Iterable, Mapping, Sequence

from env.contracts import Fill, OrderPlan
from env.fees import DEFAULT_FEE_SCHEDULE


@dataclass(frozen=True)
class AcceptedBrokerOrder:
    """Immutable identity of an order the broker accepted for submission."""

    order_id: int
    side: str
    code: str
    planned_quantity: int

    def __post_init__(self) -> None:
        if type(self.order_id) is not int or self.order_id <= 0:
            raise ValueError("accepted broker order id must be a positive int")
        if self.side not in {"buy", "sell"}:
            raise ValueError("accepted broker order side must be buy or sell")
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("accepted broker order code must be non-empty")
        if type(self.planned_quantity) is not int or self.planned_quantity <= 0:
            raise ValueError(
                "accepted broker order planned quantity must be a positive int"
            )


class BrokerExecutionError(RuntimeError):
    """Execution stopped; fields preserve every known piece of broker reality."""

    def __init__(
        self,
        message: str,
        fills: Sequence[Fill] = (),
        *,
        terminal_confirmed: bool = True,
        accepted_orders: Sequence[AcceptedBrokerOrder] = (),
    ) -> None:
        super().__init__(message)
        self.fills = tuple(fills)
        self.terminal_confirmed = bool(terminal_confirmed)
        self.accepted_orders = tuple(accepted_orders)
        if any(
            not isinstance(item, AcceptedBrokerOrder)
            for item in self.accepted_orders
        ):
            raise TypeError(
                "accepted_orders must contain AcceptedBrokerOrder values"
            )


FeeEstimator = Callable[[str, float], float]


def _broker_fee_estimate(side: str, notional: float) -> float:
    """Estimate broker charges only; adverse price movement is added later."""

    fees = DEFAULT_FEE_SCHEDULE
    value = float(notional)
    result = fees.broker_commission(value) + value * fees.transfer_fee_rate
    if side == "sell":
        result += value * fees.stamp_tax_rate
    return result


def validate_accepted_orders(
    plan: OrderPlan,
    accepted_orders: Sequence[AcceptedBrokerOrder],
) -> tuple[AcceptedBrokerOrder, ...]:
    """Bind accepted broker identities to an exact subset of ``plan``."""

    submitted = tuple(accepted_orders)
    if not submitted:
        raise ValueError("pending execution has no accepted broker orders")
    if any(not isinstance(item, AcceptedBrokerOrder) for item in submitted):
        raise TypeError("accepted_orders must contain AcceptedBrokerOrder values")
    planned = {("sell", code): quantity for code, quantity in plan.sell_orders}
    planned.update(
        (("buy", code), quantity) for code, quantity in plan.buy_orders.items()
    )
    seen_ids: set[int] = set()
    seen_intents: set[tuple[str, str]] = set()
    for item in submitted:
        intent = (item.side, item.code)
        if item.order_id in seen_ids or intent in seen_intents:
            raise ValueError("pending accepted broker orders contain duplicates")
        if planned.get(intent) != item.planned_quantity:
            raise ValueError(
                "pending accepted broker order differs from recorded OrderPlan"
            )
        seen_ids.add(item.order_id)
        seen_intents.add(intent)
    return submitted


class BrokerExecutor:
    """Submit exact planner quantities and translate broker trades to ``Fill``.

    There is deliberately no score, target-position, cash reserve, retry-sizing
    or portfolio logic here. Sells are terminally reconciled before buys are
    submitted. If submission stops midway, the exception carries the exact
    accepted-order identities needed for a later journal reconciliation.
    """

    def __init__(
        self,
        trader: object,
        *,
        buy_order_type: object,
        sell_order_type: object,
        terminal_statuses: Iterable[object],
        wait_timeout_seconds: float = 180.0,
        poll_seconds: float = 0.25,
        fee_estimator: FeeEstimator = _broker_fee_estimate,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not math.isfinite(wait_timeout_seconds) or wait_timeout_seconds < 0.0:
            raise ValueError("wait_timeout_seconds must be finite and non-negative")
        if not math.isfinite(poll_seconds) or poll_seconds < 0.0:
            raise ValueError("poll_seconds must be finite and non-negative")
        statuses = frozenset(terminal_statuses)
        if not statuses:
            raise ValueError("terminal_statuses must not be empty")
        self.trader = trader
        self.buy_order_type = buy_order_type
        self.sell_order_type = sell_order_type
        self.terminal_statuses = statuses
        self.wait_timeout_seconds = float(wait_timeout_seconds)
        self.poll_seconds = float(poll_seconds)
        self.fee_estimator = fee_estimator
        self._monotonic = monotonic
        self._sleep = sleep
        self.last_accepted_orders: tuple[AcceptedBrokerOrder, ...] = ()

    def execute(self, order_plan: OrderPlan) -> Sequence[Fill]:
        if not isinstance(order_plan, OrderPlan):
            raise TypeError("BrokerExecutor.execute requires an OrderPlan")
        self._validate_plan(order_plan)
        reference_prices = self._reference_prices(order_plan)
        submitted: list[AcceptedBrokerOrder] = []
        try:
            sells = self._submit_group(
                order_plan,
                order_plan.sell_orders,
                side="sell",
                submitted=submitted,
            )
            self._wait_terminal(sells)
            sell_fills = self._read_fills(sells, reference_prices)
            self._require_exact_fill_coverage(sells, sell_fills)

            buys = self._submit_group(
                order_plan,
                order_plan.buy_orders.items(),
                side="buy",
                submitted=submitted,
            )
            self._wait_terminal(buys)
            buy_fills = self._read_fills(buys, reference_prices)
            self._require_exact_fill_coverage(buys, buy_fills)
        except Exception as original_error:
            self.last_accepted_orders = tuple(submitted)
            cleanup_errors: list[str] = []
            try:
                terminal_confirmed = self._cancel_and_confirm_terminal(submitted)
            except Exception as cleanup_error:
                terminal_confirmed = False
                cleanup_errors.append(f"terminal check failed: {cleanup_error}")
            try:
                fills = self._read_fills(submitted, reference_prices)
            except Exception as fill_error:
                fills = (
                    original_error.fills
                    if isinstance(original_error, BrokerExecutionError)
                    else ()
                )
                terminal_confirmed = False
                cleanup_errors.append(f"final Fill read failed: {fill_error}")
            message = str(original_error)
            if cleanup_errors:
                message += "; " + "; ".join(cleanup_errors)
            raise BrokerExecutionError(
                message,
                fills,
                terminal_confirmed=terminal_confirmed,
                accepted_orders=submitted,
            ) from original_error

        self.last_accepted_orders = tuple(submitted)
        return (*sell_fills, *buy_fills)

    @staticmethod
    def _validate_plan(plan: OrderPlan) -> None:
        if not plan.decision_date:
            raise ValueError("OrderPlan decision_date must not be empty")
        sell_codes: set[str] = set()
        for code, quantity in plan.sell_orders:
            if not isinstance(code, str) or not code:
                raise ValueError("sell order code must be a non-empty string")
            if code in sell_codes:
                raise ValueError("OrderPlan contains duplicate sell codes")
            if type(quantity) is not int or quantity <= 0:
                raise ValueError("sell order quantity must be a positive int")
            sell_codes.add(code)
        for code, quantity in plan.buy_orders.items():
            if not isinstance(code, str) or not code:
                raise ValueError("buy order code must be a non-empty string")
            if type(quantity) is not int or quantity <= 0:
                raise ValueError("buy order quantity must be a positive int")

    @staticmethod
    def _reference_prices(plan: OrderPlan) -> dict[str, float]:
        raw = plan.diagnostics.get("prices")
        if not isinstance(raw, Mapping):
            raise ValueError("OrderPlan must bind per-code T-open reference prices")
        prices: dict[str, float] = {}
        required_codes = {code for code, _ in plan.sell_orders} | set(plan.buy_orders)
        for code in required_codes:
            try:
                price = float(raw[code])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"OrderPlan has no T-open reference price for {code}"
                ) from exc
            if not math.isfinite(price) or price <= 0.0:
                raise ValueError(
                    f"OrderPlan has an invalid T-open reference price for {code}"
                )
            prices[code] = price
        return prices

    def _submit_group(
        self,
        plan: OrderPlan,
        orders: Iterable[tuple[str, int]],
        *,
        side: str,
        submitted: list[AcceptedBrokerOrder],
    ) -> list[AcceptedBrokerOrder]:
        order_type = self.buy_order_type if side == "buy" else self.sell_order_type
        group: list[AcceptedBrokerOrder] = []
        for code, quantity in orders:
            order_id = self.trader.order(
                order_type,
                code,
                quantity,
                None,
                order_remark=(
                    f"canonical-plan decision={plan.decision_date} side={side}"
                ),
            )
            if type(order_id) is not int or order_id <= 0:
                raise BrokerExecutionError(
                    f"broker rejected {side} submission for {code}"
                )
            item = AcceptedBrokerOrder(
                order_id=order_id,
                code=code,
                side=side,
                planned_quantity=quantity,
            )
            submitted.append(item)
            group.append(item)
        return group

    def _wait_terminal(self, submitted: Sequence[AcceptedBrokerOrder]) -> None:
        if not submitted:
            return
        deadline = self._monotonic() + self.wait_timeout_seconds
        pending = tuple(submitted)
        while pending:
            next_pending: list[AcceptedBrokerOrder] = []
            for item in pending:
                order = self.trader.query_order(item.order_id)
                if (
                    order is None
                    or getattr(order, "order_status", None)
                    not in self.terminal_statuses
                ):
                    next_pending.append(item)
            if not next_pending:
                return
            if self._monotonic() >= deadline:
                codes = ", ".join(item.code for item in next_pending)
                raise BrokerExecutionError(
                    f"broker orders did not reach terminal state: {codes}"
                )
            self._sleep(self.poll_seconds)
            pending = tuple(next_pending)

    def _cancel_and_confirm_terminal(
        self,
        submitted: Sequence[AcceptedBrokerOrder],
    ) -> bool:
        """Best-effort cancel, then prove every accepted order is terminal."""

        if not submitted:
            return True
        cancel = getattr(self.trader, "cancel_order", None)
        for item in submitted:
            order = self.trader.query_order(item.order_id)
            if (
                order is not None
                and getattr(order, "order_status", None) in self.terminal_statuses
            ):
                continue
            if callable(cancel):
                try:
                    cancel(item.order_id)
                except Exception:
                    # A cancel can race a fill. Terminal state below is the
                    # authority; the cancel call's return is not.
                    pass
        deadline = self._monotonic() + self.wait_timeout_seconds
        while True:
            if all(
                (order := self.trader.query_order(item.order_id)) is not None
                and getattr(order, "order_status", None) in self.terminal_statuses
                for item in submitted
            ):
                return True
            if self._monotonic() >= deadline:
                return False
            self._sleep(self.poll_seconds)

    def reconcile(
        self,
        order_plan: OrderPlan,
        accepted_orders: Sequence[AcceptedBrokerOrder],
    ) -> Sequence[Fill]:
        """Reconcile only the exact broker-accepted orders recorded at failure."""

        self._validate_plan(order_plan)
        references = self._reference_prices(order_plan)
        submitted = validate_accepted_orders(order_plan, accepted_orders)
        try:
            terminal_confirmed = self._cancel_and_confirm_terminal(submitted)
            fills = self._read_fills(submitted, references)
        except Exception as error:
            raise BrokerExecutionError(
                f"pending broker reconciliation failed: {error}",
                terminal_confirmed=False,
                accepted_orders=submitted,
            ) from error
        if not terminal_confirmed:
            raise BrokerExecutionError(
                "pending broker execution is still not terminal",
                fills,
                terminal_confirmed=False,
                accepted_orders=submitted,
            )
        return fills

    def _read_fills(
        self,
        submitted: Sequence[AcceptedBrokerOrder],
        reference_prices: Mapping[str, float],
    ) -> tuple[Fill, ...]:
        if not submitted:
            return ()
        by_id = {item.order_id: item for item in submitted}
        raw_trades: list[tuple[AcceptedBrokerOrder, int, float, str]] = []
        query_all_trades = getattr(self.trader, "query_all_trades", None)
        if callable(query_all_trades):
            for trade in query_all_trades() or ():
                order_id = int(getattr(trade, "order_id", 0) or 0)
                item = by_id.get(order_id)
                if item is None:
                    continue
                quantity = int(getattr(trade, "traded_volume", 0) or 0)
                price = float(getattr(trade, "traded_price", 0.0) or 0.0)
                if quantity > 0 and math.isfinite(price) and price > 0.0:
                    raw_trades.append(
                        (item, quantity, price, self._timestamp(trade))
                    )
        represented_order_ids = {row[0].order_id for row in raw_trades}
        for item in submitted:
            if item.order_id not in represented_order_ids:
                order = self.trader.query_order(item.order_id)
                if order is None:
                    continue
                quantity = int(getattr(order, "traded_volume", 0) or 0)
                price = float(getattr(order, "traded_price", 0.0) or 0.0)
                if quantity > 0 and math.isfinite(price) and price > 0.0:
                    raw_trades.append(
                        (item, quantity, price, self._timestamp(order))
                    )
        return self._with_allocated_fees(raw_trades, reference_prices)

    def _with_allocated_fees(
        self,
        raw_trades: Sequence[tuple[AcceptedBrokerOrder, int, float, str]],
        reference_prices: Mapping[str, float],
    ) -> tuple[Fill, ...]:
        grouped: dict[
            int,
            list[tuple[AcceptedBrokerOrder, int, float, str]],
        ] = {}
        for row in raw_trades:
            grouped.setdefault(row[0].order_id, []).append(row)
        fills: list[Fill] = []
        for order_id in sorted(grouped):
            rows = grouped[order_id]
            notionals = [quantity * price for _, quantity, price, _ in rows]
            total_notional = sum(notionals)
            total_fee = float(self.fee_estimator(rows[0][0].side, total_notional))
            if not math.isfinite(total_fee) or total_fee < 0.0:
                raise BrokerExecutionError("fee estimator returned an invalid cost")
            allocated = 0.0
            for index, ((item, quantity, price, timestamp), notional) in enumerate(
                zip(rows, notionals, strict=True)
            ):
                fee = (
                    total_fee - allocated
                    if index == len(rows) - 1
                    else total_fee * notional / total_notional
                )
                allocated += fee
                reference_price = float(reference_prices[item.code])
                adverse_price_move = (
                    max(0.0, price - reference_price)
                    if item.side == "buy"
                    else max(0.0, reference_price - price)
                )
                fills.append(
                    Fill(
                        code=item.code,
                        side=item.side,
                        quantity=quantity,
                        price=price,
                        fee=fee + adverse_price_move * quantity,
                        timestamp=timestamp,
                    )
                )
        return tuple(fills)

    @staticmethod
    def _require_exact_fill_coverage(
        submitted: Sequence[AcceptedBrokerOrder],
        fills: Sequence[Fill],
    ) -> None:
        actual: dict[tuple[str, str], int] = {}
        for fill in fills:
            key = (fill.side, fill.code)
            actual[key] = actual.get(key, 0) + fill.quantity
        missing: list[str] = []
        for item in submitted:
            filled = actual.get((item.side, item.code), 0)
            if filled != item.planned_quantity:
                missing.append(
                    f"{item.side} {item.code}: "
                    f"planned={item.planned_quantity}, filled={filled}"
                )
        if missing:
            raise BrokerExecutionError(
                "broker fill coverage differs from canonical plan; "
                + "; ".join(missing),
                fills,
            )

    @staticmethod
    def _timestamp(value: object) -> str:
        raw = getattr(value, "traded_time", None)
        if raw is None:
            raw = getattr(value, "timestamp", None)
        if (
            isinstance(raw, (int, float))
            and math.isfinite(float(raw))
            and raw > 0
        ):
            return datetime.fromtimestamp(float(raw)).isoformat(timespec="seconds")
        return "" if raw is None else str(raw)


__all__ = [
    "AcceptedBrokerOrder",
    "BrokerExecutionError",
    "BrokerExecutor",
    "FeeEstimator",
    "validate_accepted_orders",
]
