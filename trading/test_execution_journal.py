"""Fake broker only: process loss before/after submission must not duplicate targets."""
from datetime import date
from types import SimpleNamespace as NS

import pytest
from xtquant import xtconstant as xc

from trading.execution_journal import ExecutionJournal
from trading.executor import OrderMonitor, RebalanceExecutor


DAY = date(2026, 9, 4)


def plan():
    return dict(signal_date=DAY, trade_date=DAY, sell_orders=[],
                buy_allocations={'600000.SH': 100}, buy_n_stocks=['600000.SH'],
                prices={'600000.SH': 10.0}, limit_prices={})


class Broker:
    strategy_name = 'FlashMan'

    def __init__(self):
        self.orders = []
        self.crash_after_accept = False

    def query_orders(self):
        return self.orders

    def query_asset(self):
        return NS(cash=100000)

    def query_order(self, oid):
        return next(o for o in self.orders if o.order_id == oid)

    def order(self, order_type, code, shares, price, order_remark):
        order = NS(order_id=len(self.orders) + 1, order_type=order_type,
                   stock_code=code, order_volume=shares, traded_volume=shares,
                   order_remark=order_remark, order_status=xc.ORDER_SUCCEEDED,
                   status_msg='', strategy_name=self.strategy_name)
        self.orders.append(order)
        if self.crash_after_accept:
            raise ConnectionError('accepted, response lost')
        return order.order_id


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    import trading.executor as module
    monkeypatch.setattr(OrderMonitor, '_can_submit_now', lambda *a: True)
    monkeypatch.setattr(OrderMonitor, 'record_action', lambda *a, **k: None)
    monkeypatch.setattr(module.live_trade_recorder, 'record_event', lambda *a, **k: None)
    monkeypatch.setattr(RebalanceExecutor, '_summarize', lambda *a: None)
    # Execute actual submission/target accounting without waiting on wall-clock sessions.
    monkeypatch.setattr(OrderMonitor, '_main_loop', lambda self: self._submit_affordable_buys())


def test_completed_restart_never_resubmits(tmp_path):
    broker = Broker()
    for _ in range(2):
        RebalanceExecutor(broker, journal=ExecutionJournal(tmp_path, 'test')).execute(plan())
    assert len(broker.orders) == 1


def test_response_lost_after_accept_recovers_broker_fill(tmp_path):
    broker = Broker()
    broker.crash_after_accept = True
    with pytest.raises(ConnectionError):
        RebalanceExecutor(broker, journal=ExecutionJournal(tmp_path, 'test')).execute(plan())
    broker.crash_after_accept = False
    RebalanceExecutor(broker, journal=ExecutionJournal(tmp_path, 'test')).execute(plan())
    assert len(broker.orders) == 1


def test_unconfirmed_intent_blocks_resubmit(tmp_path):
    journal = ExecutionJournal(tmp_path, 'test')
    journal.begin(plan())
    journal.before_submit('BUY', '600000.SH', 100, 10.0)
    broker = Broker()
    with pytest.raises(RuntimeError, match='状态不确定'):
        RebalanceExecutor(broker, journal=ExecutionJournal(tmp_path, 'test')).execute(plan())
    assert broker.orders == []


def test_existing_legacy_broker_order_requires_reconciliation(tmp_path):
    broker = Broker()
    broker.order(xc.STOCK_BUY, '600000.SH', 100, 10, 'legacy')
    with pytest.raises(RuntimeError, match='缺执行记录'):
        RebalanceExecutor(broker, journal=ExecutionJournal(tmp_path, 'test')).execute(plan())
    assert len(broker.orders) == 1


def test_recovery_counts_partial_fill_and_submits_only_remainder(tmp_path):
    broker = Broker()
    journal = ExecutionJournal(tmp_path, 'test')
    pending = plan()
    pending['buy_allocations']['600000.SH'] = 300
    journal.begin(pending)
    intent = journal.before_submit('BUY', '600000.SH', 300, 10)
    broker.order(xc.STOCK_BUY, '600000.SH', 300, 10, intent['remark'])
    broker.orders[0].traded_volume = 100
    broker.orders[0].order_status = xc.ORDER_PART_CANCEL
    # The first iteration acknowledges the canceled order; the next can retry its remainder.
    def run(self):
        self._filled_inflight('600000.SH')
        self._handle_terminal_orders()
        self._submit_affordable_buys()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(OrderMonitor, '_main_loop', run)
        RebalanceExecutor(broker, journal=ExecutionJournal(tmp_path, 'test')).execute(pending)
    assert [o.order_volume for o in broker.orders] == [300, 200]


def test_full_sell_target_is_fixed_before_partial_execution(tmp_path):
    broker = Broker()
    broker.query_account_snapshot = lambda: (broker.query_asset(), [NS(stock_code='600000.SH', can_use_volume=300)])
    broker.crash_after_accept = True
    pending = plan()
    pending.update(sell_orders=[('600000.SH', -1)], buy_allocations={}, buy_n_stocks=[])
    with pytest.raises(ConnectionError):
        RebalanceExecutor(broker, journal=ExecutionJournal(tmp_path, 'test')).execute(pending)
    assert ExecutionJournal(tmp_path, 'test').load(DAY)['sell_orders'] == [['600000.SH', 300]]


def test_journal_write_failure_prevents_first_order(tmp_path, monkeypatch):
    journal = ExecutionJournal(tmp_path, 'test')
    journal.begin(plan())
    def failed():
        raise OSError('disk full')
    monkeypatch.setattr(journal, 'save', failed)
    broker = Broker()
    with pytest.raises(OSError):
        RebalanceExecutor(broker, journal=journal).submit_order('BUY', '600000.SH', 100, 10, DAY, DAY)
    assert broker.orders == []


@pytest.mark.parametrize('field,value', [('order_type', xc.STOCK_SELL), ('stock_code', '600001.SH'), ('order_volume', 200)])
def test_wrong_broker_match_is_rejected(tmp_path, field, value):
    journal = ExecutionJournal(tmp_path, 'test')
    journal.begin(plan())
    intent = journal.before_submit('BUY', '600000.SH', 100, 10)
    broker = Broker()
    broker.order(xc.STOCK_BUY, '600000.SH', 100, 10, intent['remark'])
    setattr(broker.orders[0], field, value)
    with pytest.raises(RuntimeError, match='不一致'):
        journal.recover_orders(broker.orders)


def test_unfilled_target_is_not_completed_and_can_resume(tmp_path):
    broker = Broker()
    broker.query_asset = lambda: NS(cash=0)
    RebalanceExecutor(broker, journal=ExecutionJournal(tmp_path, 'test')).execute(plan())
    journal = ExecutionJournal(tmp_path, 'test')
    journal.load(DAY)
    assert journal.state['completed'] is False
    assert broker.orders == []
    broker.query_asset = lambda: NS(cash=100000)
    RebalanceExecutor(broker, journal=journal).execute(plan())
    assert len(broker.orders) == 1
    assert journal.state['completed'] is True


def test_restart_retains_original_open_limit_deadline(tmp_path):
    broker = Broker()
    broker.crash_after_accept = True
    first = RebalanceExecutor(broker, journal=ExecutionJournal(tmp_path, 'test'))
    with pytest.raises(ConnectionError):
        first.execute(plan())
    deadline = first.journal.state['open_limit_deadline']
    second = RebalanceExecutor(broker, journal=ExecutionJournal(tmp_path, 'test'))
    broker.crash_after_accept = False
    second.execute(plan())
    assert second._monitor.open_limit_deadline == deadline
