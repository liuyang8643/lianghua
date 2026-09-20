from types import SimpleNamespace

import pytest

from env.contracts import OrderPlan
from trade.executor import (
    AcceptedBrokerOrder,
    BrokerExecutionError,
    BrokerExecutor,
)


class FakeTrader:
    def __init__(self):
        self.calls = []
        self.orders = {}
        self.trades = []

    def order(self, order_type, code, quantity, price, order_remark=""):
        order_id = len(self.calls) + 1
        self.calls.append((order_type, code, quantity, price, order_remark))
        fill_price = 10.0 + order_id
        self.orders[order_id] = SimpleNamespace(
            order_status="done",
            traded_volume=quantity,
            traded_price=fill_price,
        )
        self.trades.append(
            SimpleNamespace(
                order_id=order_id,
                traded_volume=quantity,
                traded_price=fill_price,
                traded_time=1_700_000_000 + order_id,
            )
        )
        return order_id

    def query_order(self, order_id):
        return self.orders[order_id]

    def query_all_trades(self):
        return list(self.trades)


def test_broker_submits_exact_plan_quantities_sell_then_buy():
    trader = FakeTrader()
    executor = BrokerExecutor(
        trader,
        buy_order_type="BUY",
        sell_order_type="SELL",
        terminal_statuses={"done"},
        wait_timeout_seconds=0.0,
        fee_estimator=lambda side, notional: 0.0,
    )
    plan = OrderPlan(
        decision_date="2026-08-28",
        sell_orders=(("600001.SH", 300), ("000001.SZ", 200)),
        buy_orders={"300001.SZ": 500},
        diagnostics={
            "prices": {
                "600001.SH": 11.0,
                "000001.SZ": 12.0,
                "300001.SZ": 13.0,
            }
        },
    )

    fills = tuple(executor.execute(plan))

    assert [(call[0], call[1], call[2]) for call in trader.calls] == [
        ("SELL", "600001.SH", 300),
        ("SELL", "000001.SZ", 200),
        ("BUY", "300001.SZ", 500),
    ]
    assert [(fill.side, fill.code, fill.quantity) for fill in fills] == [
        ("sell", "600001.SH", 300),
        ("sell", "000001.SZ", 200),
        ("buy", "300001.SZ", 500),
    ]
    assert [item.order_id for item in executor.last_accepted_orders] == [1, 2, 3]


def test_broker_rejects_legacy_pending_mapping():
    executor = BrokerExecutor(
        FakeTrader(),
        buy_order_type="BUY",
        sell_order_type="SELL",
        terminal_statuses={"done"},
    )
    with pytest.raises(TypeError, match="OrderPlan"):
        executor.execute({"buy_allocations": {"300001.SZ": 500}})


class PartialSellTrader(FakeTrader):
    def order(self, order_type, code, quantity, price, order_remark=""):
        order_id = super().order(order_type, code, quantity, price, order_remark)
        if order_type == "SELL":
            self.orders[order_id].traded_volume = 100
            self.trades[-1].traded_volume = 100
        return order_id


def test_partial_terminal_sell_is_journalable_and_blocks_all_buys():
    trader = PartialSellTrader()
    executor = BrokerExecutor(
        trader,
        buy_order_type="BUY",
        sell_order_type="SELL",
        terminal_statuses={"done"},
        wait_timeout_seconds=0.0,
        fee_estimator=lambda side, notional: 0.0,
    )
    plan = OrderPlan(
        decision_date="2026-08-28",
        sell_orders=(("600001.SH", 300),),
        buy_orders={"300001.SZ": 500},
        diagnostics={"prices": {"600001.SH": 11.0, "300001.SZ": 12.0}},
    )

    with pytest.raises(BrokerExecutionError, match="fill coverage") as caught:
        executor.execute(plan)

    assert [(call[0], call[1]) for call in trader.calls] == [
        ("SELL", "600001.SH")
    ]
    assert [(fill.side, fill.quantity) for fill in caught.value.fills] == [
        ("sell", 100)
    ]


class RejectSecondSellTrader(FakeTrader):
    def order(self, order_type, code, quantity, price, order_remark=""):
        if len(self.calls) == 1:
            self.calls.append((order_type, code, quantity, price, order_remark))
            return 0
        return super().order(order_type, code, quantity, price, order_remark)


def test_mid_group_rejection_keeps_first_submitted_order_reality():
    trader = RejectSecondSellTrader()
    executor = BrokerExecutor(
        trader,
        buy_order_type="BUY",
        sell_order_type="SELL",
        terminal_statuses={"done"},
        wait_timeout_seconds=0.0,
        fee_estimator=lambda side, notional: 0.0,
    )
    plan = OrderPlan(
        decision_date="2026-08-28",
        sell_orders=(("600001.SH", 300), ("000001.SZ", 200)),
        diagnostics={"prices": {"600001.SH": 11.0, "000001.SZ": 12.0}},
    )

    with pytest.raises(BrokerExecutionError, match="rejected") as caught:
        executor.execute(plan)

    assert [item.order_id for item in executor.last_accepted_orders] == [1]
    assert [(fill.code, fill.quantity) for fill in caught.value.fills] == [
        ("600001.SH", 300)
    ]


class PendingThenFilledTrader(RejectSecondSellTrader):
    def order(self, order_type, code, quantity, price, order_remark=""):
        order_id = super().order(order_type, code, quantity, price, order_remark)
        if order_id > 0:
            self.orders[order_id].order_status = "pending"
            self.orders[order_id].traded_volume = 0
            self.trades.clear()
        return order_id

    def cancel_order(self, order_id):
        order = self.orders[order_id]
        order.order_status = "done"
        order.traded_volume = 300
        order.traded_price = 11.0
        self.trades.append(
            SimpleNamespace(
                order_id=order_id,
                traded_volume=300,
                traded_price=11.0,
                traded_time=1_700_000_100,
            )
        )


def test_mid_group_reject_waits_for_pending_predecessor_late_fill():
    trader = PendingThenFilledTrader()
    executor = BrokerExecutor(
        trader,
        buy_order_type="BUY",
        sell_order_type="SELL",
        terminal_statuses={"done", "canceled"},
        wait_timeout_seconds=0.0,
        fee_estimator=lambda side, notional: 0.0,
    )
    plan = OrderPlan(
        decision_date="2026-08-28",
        sell_orders=(("600001.SH", 300), ("000001.SZ", 200)),
        buy_orders={"300001.SZ": 500},
        diagnostics={
            "prices": {
                "600001.SH": 11.0,
                "000001.SZ": 12.0,
                "300001.SZ": 13.0,
            }
        },
    )

    with pytest.raises(BrokerExecutionError) as caught:
        executor.execute(plan)

    assert caught.value.terminal_confirmed is True
    assert [(fill.code, fill.quantity) for fill in caught.value.fills] == [
        ("600001.SH", 300)
    ]
    assert all(call[0] == "SELL" for call in trader.calls)


class PendingThenCanceledTrader(PendingThenFilledTrader):
    def cancel_order(self, order_id):
        order = self.orders[order_id]
        order.order_status = "canceled"
        order.traded_volume = 0
        order.traded_price = 0.0


def test_mid_group_reject_can_finalize_cancel_confirmed_zero_fill():
    trader = PendingThenCanceledTrader()
    executor = BrokerExecutor(
        trader,
        buy_order_type="BUY",
        sell_order_type="SELL",
        terminal_statuses={"done", "canceled"},
        wait_timeout_seconds=0.0,
        fee_estimator=lambda side, notional: 0.0,
    )
    plan = OrderPlan(
        decision_date="2026-08-28",
        sell_orders=(("600001.SH", 300), ("000001.SZ", 200)),
        diagnostics={"prices": {"600001.SH": 11.0, "000001.SZ": 12.0}},
    )

    with pytest.raises(BrokerExecutionError) as caught:
        executor.execute(plan)

    assert caught.value.terminal_confirmed is True
    assert caught.value.fills == ()


class NeverTerminalTrader(PendingThenFilledTrader):
    def cancel_order(self, order_id):
        return None


def test_unconfirmed_cancel_timeout_is_explicitly_unsettled():
    trader = NeverTerminalTrader()
    executor = BrokerExecutor(
        trader,
        buy_order_type="BUY",
        sell_order_type="SELL",
        terminal_statuses={"done", "canceled"},
        wait_timeout_seconds=0.0,
        fee_estimator=lambda side, notional: 0.0,
    )
    plan = OrderPlan(
        decision_date="2026-08-28",
        sell_orders=(("600001.SH", 300), ("000001.SZ", 200)),
        diagnostics={"prices": {"600001.SH": 11.0, "000001.SZ": 12.0}},
    )

    with pytest.raises(BrokerExecutionError) as caught:
        executor.execute(plan)

    assert caught.value.terminal_confirmed is False
    assert caught.value.accepted_orders == (
        AcceptedBrokerOrder(
            order_id=1,
            side="sell",
            code="600001.SH",
            planned_quantity=300,
        ),
    )
    assert caught.value.fills == ()


def test_pending_subset_can_reconcile_a_later_fill_without_full_plan_ids():
    trader = NeverTerminalTrader()
    executor = BrokerExecutor(
        trader,
        buy_order_type="BUY",
        sell_order_type="SELL",
        terminal_statuses={"done", "canceled"},
        wait_timeout_seconds=0.0,
        fee_estimator=lambda side, notional: 0.0,
    )
    plan = OrderPlan(
        decision_date="2026-08-28",
        sell_orders=(("600001.SH", 300), ("000001.SZ", 200)),
        diagnostics={"prices": {"600001.SH": 11.0, "000001.SZ": 12.0}},
    )
    with pytest.raises(BrokerExecutionError) as caught:
        executor.execute(plan)

    trader.orders[1].order_status = "done"
    trader.orders[1].traded_volume = 300
    trader.orders[1].traded_price = 11.0
    trader.trades.append(
        SimpleNamespace(
            order_id=1,
            traded_volume=300,
            traded_price=11.0,
            traded_time=1_700_000_101,
        )
    )

    fills = tuple(executor.reconcile(plan, caught.value.accepted_orders))

    assert [(fill.code, fill.quantity) for fill in fills] == [
        ("600001.SH", 300)
    ]


class DisconnectDuringCleanupTrader(RejectSecondSellTrader):
    def __init__(self):
        super().__init__()
        self.query_count = 0

    def query_order(self, order_id):
        self.query_count += 1
        if self.query_count > 1:
            raise ConnectionError("broker disconnected during cleanup")
        return super().query_order(order_id)


def test_cleanup_disconnect_is_wrapped_with_accepted_order_identity():
    trader = DisconnectDuringCleanupTrader()
    executor = BrokerExecutor(
        trader,
        buy_order_type="BUY",
        sell_order_type="SELL",
        terminal_statuses={"done"},
        wait_timeout_seconds=0.0,
        fee_estimator=lambda side, notional: 0.0,
    )
    plan = OrderPlan(
        decision_date="2026-08-28",
        sell_orders=(("600001.SH", 300), ("000001.SZ", 200)),
        diagnostics={"prices": {"600001.SH": 11.0, "000001.SZ": 12.0}},
    )

    with pytest.raises(BrokerExecutionError, match="terminal check failed") as caught:
        executor.execute(plan)

    assert caught.value.terminal_confirmed is False
    assert caught.value.accepted_orders == (
        AcceptedBrokerOrder(1, "sell", "600001.SH", 300),
    )
    assert isinstance(caught.value.__cause__, BrokerExecutionError)


def test_fill_fee_includes_broker_fee_and_adverse_t_open_slippage_once():
    trader = FakeTrader()
    executor = BrokerExecutor(
        trader,
        buy_order_type="BUY",
        sell_order_type="SELL",
        terminal_statuses={"done"},
        wait_timeout_seconds=0.0,
        fee_estimator=lambda side, notional: 2.0,
    )
    plan = OrderPlan(
        decision_date="2026-08-28",
        buy_orders={"300001.SZ": 100},
        diagnostics={"prices": {"300001.SZ": 10.0}},
    )

    fills = tuple(executor.execute(plan))

    # FakeTrader's first actual fill is 11.0: (11-10)*100 adverse slippage,
    # plus the injected 2.0 broker fee. Journal must not add it again.
    assert fills[0].price == 11.0
    assert fills[0].fee == pytest.approx(102.0)
