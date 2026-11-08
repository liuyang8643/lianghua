from datetime import date, datetime
import json
from types import SimpleNamespace
import time

import pytest

from trading.bridge_client import BridgeClient, BridgeUnavailable
from trading.bridge_wire import open_database
from trading.qmt_bridge import BrokerBridge, UncertainOrder
from trading.bridge_wire import ConfirmedNotSubmitted


class Broker:
    def __init__(self):
        self.orders = []
        self.submissions = []
        self.cancellations = []
        self.raise_after_submit = False

    def query_stock_orders(self, account, cancelable_only=False):
        return self.orders

    def query_stock_asset(self, account):
        return SimpleNamespace(cash=10000.0, total_asset=11000.0, market_value=1000.0)

    def order_stock(self, account, code, direction, volume, price_type, price, strategy, remark):
        self.submissions.append((code, direction, volume, remark))
        if self.raise_after_submit:
            raise RuntimeError('Disconnected after broker submission')
        return 123

    def cancel_order_stock(self, account, order_id):
        self.cancellations.append(order_id)
        return 0


@pytest.fixture
def bridge(tmp_path):
    broker = Broker()
    config = {'root': str(tmp_path), 'account': 'test-account',
              'live_enabled': True, 'strategy_name': 'FlashMan',
              'price_types': [11, 44], 'trading_dates': ['2026-11-02']}
    db = open_database(tmp_path / 'bridge.sqlite3')
    server = BrokerBridge(config, broker, object(), db,
                          clock=lambda: datetime(2026, 11, 2, 10).timestamp(),
                          owner_check=lambda pid: None, instance='instance-a')
    yield server, broker, tmp_path
    db.close()


def intent(**changes):
    value = dict(account_id='test-account', trade_date='2026-11-02',
                 stock_code='688092.SH', order_type=23, volume=200,
                 price_type=11, price=20.0, strategy_name='FlashMan',
                 order_remark='wbr:abcdef123456:000001')
    value.update(changes)
    return value


def test_same_intent_across_request_ids_never_submits_twice(bridge):
    server, broker, _ = bridge
    assert server.dispatch('submit', intent(), 1) == 123
    assert server.dispatch('submit', intent(), 1) == 123
    assert len(broker.submissions) == 1
    with pytest.raises(RuntimeError, match='different intent'):
        server.dispatch('submit', intent(volume=300), 1)


@pytest.mark.parametrize('returned', [0, -1])
@pytest.mark.parametrize('delayed', [False, True])
def test_nonpositive_return_uses_real_order_and_never_resends(bridge, returned, delayed):
    server, broker, _ = bridge
    payload = intent()
    actual = SimpleNamespace(order_remark=payload['order_remark'], stock_code=payload['stock_code'],
        order_volume=payload['volume'], order_type=payload['order_type'],
        strategy_name=payload['strategy_name'], order_id=789)
    def submit(*args):
        broker.submissions.append(args)
        if not delayed:
            broker.orders = [actual]
        return returned
    broker.order_stock = submit
    if delayed:
        with pytest.raises(UncertainOrder):
            server.dispatch('submit', payload, 1)
        assert server.database.execute('SELECT state FROM order_intents').fetchone()[0] == 'executing'
        with pytest.raises(UncertainOrder):
            server.dispatch('submit', payload, 1)
        broker.orders = [actual]
    assert server.dispatch('submit', payload, 1) == 789
    assert server.dispatch('submit', payload, 1) == 789
    assert len(broker.submissions) == 1


@pytest.mark.parametrize('change', [dict(stock_code='000001.SZ'), dict(order_volume=300),
    dict(order_type=24), dict(strategy_name='manual'), dict(order_id=0), {'duplicate': True}])
def test_zero_return_with_conflicting_broker_fact_blocks(bridge, change):
    server, broker, _ = bridge
    payload = intent()
    def submit(*args):
        fields = dict(order_remark=payload['order_remark'], stock_code=payload['stock_code'],
            order_volume=payload['volume'], order_type=payload['order_type'],
            strategy_name=payload['strategy_name'], order_id=789)
        fields.update({k:v for k,v in change.items() if k != 'duplicate'})
        broker.orders = [SimpleNamespace(**fields)] * (2 if change.get('duplicate') else 1)
        return 0
    broker.order_stock = submit
    with pytest.raises(UncertainOrder):
        server.dispatch('submit', payload, 1)
    assert server.database.execute('SELECT state FROM order_intents').fetchone()[0] == 'executing'


def test_disconnect_during_submit_never_resends_when_cache_is_empty(bridge):
    server, broker, _ = bridge
    broker.raise_after_submit = True
    with pytest.raises(RuntimeError, match='Disconnected'):
        server.dispatch('submit', intent(), 1)
    assert server.database.execute('SELECT state FROM order_intents').fetchone()[0] == 'executing'
    broker.raise_after_submit = False
    with pytest.raises(UncertainOrder, match='unknown'):
        server.dispatch('submit', intent(), 1)
    assert len(broker.submissions) == 1


def test_interrupted_intent_recovers_matching_broker_order_without_resend(bridge):
    server, broker, _ = bridge
    broker.raise_after_submit = True
    with pytest.raises(RuntimeError):
        server.dispatch('submit', intent(), 1)
    broker.orders = [SimpleNamespace(order_remark=intent()['order_remark'],
                    stock_code='688092.SH', order_volume=200, order_type=23,
                    strategy_name='FlashMan', order_id=456)]
    assert server.dispatch('submit', intent(), 1) == 456
    assert len(broker.submissions) == 1


@pytest.mark.parametrize('change', [dict(account_id='other'), dict(volume=0),
    dict(volume=-100), dict(volume=True), dict(order_type=1),
    dict(strategy_name='other'), dict(order_remark='arbitrary'), dict(price=float('nan')),
    dict(trade_date='2026-10-30'), dict(stock_code='invalid')])
def test_invalid_trade_never_reaches_broker(bridge, change):
    server, broker, _ = bridge
    with pytest.raises((RuntimeError, ValueError)):
        server.dispatch('submit', intent(**change), 1)
    assert not broker.submissions


def test_closed_market_readonly_and_halt_each_prohibit_submission(bridge):
    server, broker, folder = bridge
    server.clock = lambda: datetime(2026, 11, 2, 9, 29, 59).timestamp()
    with pytest.raises(RuntimeError, match='outside'):
        server.dispatch('submit', intent(), 1)
    server.clock = lambda: datetime(2026, 11, 2, 10).timestamp()
    server.config['live_enabled'] = False
    with pytest.raises(RuntimeError, match='read-only'):
        server.dispatch('submit', intent(order_remark='wbr:abcdef123456:000002'), 1)
    server.config['live_enabled'] = True
    halt = folder / 'data/live_trades/trading_watchdog.halt'
    halt.parent.mkdir(parents=True)
    halt.touch()
    with pytest.raises(RuntimeError, match='halt'):
        server.dispatch('submit', intent(order_remark='wbr:abcdef123456:000003'), 1)
    assert not broker.submissions


def test_only_confirmed_owned_order_can_be_cancelled(bridge):
    server, broker, _ = bridge
    broker.orders = [SimpleNamespace(order_id=123, strategy_name='another',
                                    order_remark=intent()['order_remark'], order_status=50,
                                    stock_code='688092.SH', order_type=23, order_volume=200)]
    with pytest.raises(ValueError, match='unowned'):
        server.dispatch('cancel', intent(order_id=123), 1)
    assert not broker.cancellations
    broker.orders[0].strategy_name = 'FlashMan'
    with pytest.raises(ValueError, match='unowned'):
        server.dispatch('cancel', intent(order_id=123), 1)
    server.database.execute('INSERT INTO order_intents VALUES (?,?,?,?)',
                            (intent()['order_remark'], json.dumps(intent()), 'done', 123))
    server.database.commit()
    assert server.dispatch('cancel', intent(order_id=123), 1) == 0
    assert broker.cancellations == [123]


@pytest.mark.parametrize('expires,instance', [(0, 'instance-a'), (9999999999, 'old-instance')])
def test_expired_or_prior_instance_request_never_executes(bridge, expires, instance):
    server, broker, _ = bridge
    server.database.execute('INSERT INTO requests VALUES (?,?,?,?,?,?,?,?,NULL)',
        ('request', 0, expires, 1, instance, 'submit', json.dumps(intent()), 'pending'))
    server.database.commit()
    assert server.run_one()
    state, response = server.database.execute('SELECT state,response FROM requests').fetchone()
    assert state == 'done' and not json.loads(response)['ok']
    assert not broker.submissions


def test_real_sqlite_request_response_roundtrip_preserves_asset_fields(bridge, monkeypatch):
    server, broker, folder = bridge
    server.clock = time.time
    client = BridgeClient('test-account', folder)
    monkeypatch.setattr(client, '_heartbeat', lambda: dict(
        instance='instance-a', live_enabled=True))
    client.start()
    client.connect()
    monkeypatch.setattr('trading.bridge_client.time.sleep', lambda seconds: server.run_one())
    try:
        asset = client.query_stock_asset(None)
        assert asset.cash == 10000.0 and asset.market_value == 1000.0
    finally:
        client.stop()


@pytest.mark.parametrize('race', ['deadline', 'close', 'halt'])
@pytest.mark.parametrize('operation', ['submit', 'cancel'])
def test_slow_broker_query_cannot_cross_a_trade_guard(bridge, race, operation):
    server, broker, folder = bridge
    clock = [datetime(2026, 11, 2, 14, 59, 59).timestamp()]
    server.clock = lambda: clock[0]
    server.request_deadline = clock[0] + 1
    broker.orders = [] if operation == 'submit' else [SimpleNamespace(
        order_id=123, strategy_name='FlashMan', order_remark=intent()['order_remark'],
        order_status=50, stock_code='688092.SH', order_type=23, order_volume=200)]
    if operation == 'cancel':
        server.database.execute('INSERT INTO order_intents VALUES (?,?,?,?)',
                                (intent()['order_remark'], json.dumps(intent()), 'done', 123))
        server.database.commit()
    def slow_orders(account, cancelable_only=False):
        if race == 'deadline':
            clock[0] += 1.1
        elif race == 'close':
            server.request_deadline += 60
            clock[0] += 2
        else:
            halt = folder / 'data/live_trades/trading_watchdog.halt'
            halt.parent.mkdir(parents=True)
            halt.touch()
        return broker.orders
    broker.query_stock_orders = slow_orders
    with pytest.raises(RuntimeError):
        server.dispatch(operation, intent(order_id=123), 1)
    assert not broker.submissions and not broker.cancellations


def test_bridge_instance_change_escapes_scheduler_exception_handlers(bridge, monkeypatch):
    _, _, folder = bridge
    client = BridgeClient('test-account', folder)
    client.instance = 'old-instance'
    monkeypatch.setattr(client, '_heartbeat', lambda: dict(instance='new-instance', live_enabled=True))
    # Scheduler catches Exception; this must exit main and rebuild Trader through watchdog.
    with pytest.raises(BridgeUnavailable):
        try:
            client.query_stock_asset(None)
        except Exception:
            pytest.fail('Bridge restart was swallowed by scheduler-style error handling')


def test_heartbeat_uses_process_birth_not_later_reconnect_time(bridge, monkeypatch):
    _, _, folder = bridge
    client = BridgeClient('test-account', folder)
    state = dict(connected=True, account='test-account', updated=time.time(),
                 source_sha256=client.expected_source_hash, pid=101, parent_pid=202,
                 executable='C:/broker/pythonw.exe', parent_image='C:/broker/miniquote.exe',
                 started=100.0, session_started=1000.0, instance='reconnected', live_enabled=True)
    (folder / 'heartbeat.json').write_text(json.dumps(state), encoding='utf-8')
    process = SimpleNamespace(exe=lambda: state['executable'], ppid=lambda: 202,
                              create_time=lambda: 100.0)
    parent = SimpleNamespace(pid=202, exe=lambda: state['parent_image'])
    monkeypatch.setattr('trading.bridge_client.psutil.Process',
                        lambda pid: process if pid == 101 else parent)
    assert client._heartbeat()['instance'] == 'reconnected'
    state['started'] = state['session_started']
    (folder / 'heartbeat.json').write_text(json.dumps(state), encoding='utf-8')
    with pytest.raises(RuntimeError, match='identity'):
        client._heartbeat()


@pytest.mark.parametrize('lost_local_ack', [False, True])
def test_confirmed_no_submission_recovers_journal_and_retries_original_target(bridge, lost_local_ack):
    from trading.execution_journal import ExecutionJournal
    server, broker, folder = bridge
    day = date(2026, 11, 2)
    plan = dict(signal_date=day, trade_date=day, sell_orders=[],
                buy_allocations={'688092.SH': 200}, buy_n_stocks=['688092.SH'],
                prices={'688092.SH': 20.0}, limit_prices={})
    journal = ExecutionJournal(folder / 'journal', 'test-account')
    journal.begin(plan)
    request = journal.before_submit('BUY', '688092.SH', 200, 20.0)
    clock = [datetime(2026, 11, 2, 11, 29, 59).timestamp()]
    server.clock = lambda: clock[0]
    def slow_query(account, cancelable_only=False):
        clock[0] += 2
        return []
    broker.query_stock_orders = slow_query
    with pytest.raises(ConfirmedNotSubmitted) as rejected:
        server.dispatch('submit', intent(order_remark=request['remark']), 1)
    assert not broker.submissions
    if not lost_local_ack:
        journal.mark_not_submitted(request, rejected.value)
    restored = ExecutionJournal(folder / 'journal', 'test-account')
    pending = restored.load(day)
    proof = server.dispatch('unsubmitted_intents',
                            dict(account_id='test-account', requests=restored.state['requests']), 1)
    assert restored.recover_orders([], not_submitted=proof) == []
    assert pending['buy_allocations'] == {'688092.SH': 200}
    next_request = restored.before_submit('BUY', '688092.SH', 200, 20.0)
    assert next_request['remark'] != request['remark']
    clock[0] = datetime(2026, 11, 2, 13, 0, 1).timestamp()
    broker.query_stock_orders = lambda account, cancelable_only=False: []
    assert server.dispatch('submit', intent(order_remark=next_request['remark']), 1) == 123
    assert len(broker.submissions) == 1


def test_unknown_submission_never_gets_a_no_submission_proof(bridge):
    from trading.execution_journal import ExecutionJournal
    server, broker, folder = bridge
    journal = ExecutionJournal(folder / 'journal', 'test-account')
    day = date(2026, 11, 2)
    journal.begin(dict(signal_date=day, trade_date=day, sell_orders=[],
                       buy_allocations={'688092.SH': 200}, buy_n_stocks=['688092.SH'],
                       prices={'688092.SH': 20.0}, limit_prices={}))
    request = journal.before_submit('BUY', '688092.SH', 200, 20.0)
    broker.raise_after_submit = True
    with pytest.raises(RuntimeError):
        server.dispatch('submit', intent(order_remark=request['remark']), 1)
    proof = server.dispatch('unsubmitted_intents',
                            dict(account_id='test-account', requests=journal.state['requests']), 1)
    assert proof == []
    with pytest.raises(RuntimeError, match='状态不确定'):
        journal.recover_orders([], not_submitted=proof)
    assert len(broker.submissions) == 1


def test_rpc_proof_reaches_executor_and_is_saved_before_retry(bridge, monkeypatch):
    from trading.execution_journal import ExecutionJournal
    from trading.executor import RebalanceExecutor
    server, broker, folder = bridge
    server.clock = time.time
    client = BridgeClient('test-account', folder)
    monkeypatch.setattr(client, '_heartbeat', lambda: dict(instance='instance-a', live_enabled=True))
    client.start()
    client.connect()
    monkeypatch.setattr('trading.bridge_client.time.sleep', lambda seconds: server.run_one())
    # Force a known pre-submission rejection without making a fake broker call.
    server.config['live_enabled'] = False
    day = date.today()
    journal = ExecutionJournal(folder / 'journal', 'test-account')
    journal.begin(dict(signal_date=day, trade_date=day, sell_orders=[],
                       buy_allocations={'688092.SH': 200}, buy_n_stocks=['688092.SH'],
                       prices={'688092.SH': 20.0}, limit_prices={}))
    def order(direction, code, shares, price, order_remark):
        return client.order_stock(None, code, direction, shares, 11, price, 'FlashMan', order_remark)
    executor = RebalanceExecutor(SimpleNamespace(order=order), journal=journal)
    try:
        assert executor.submit_order('BUY', '688092.SH', 200, 20.0, day, day) is None
        restored = ExecutionJournal(folder / 'journal', 'test-account')
        restored.load(day)
        assert restored.state['requests'][0]['submission_state'] == 'not_submitted'
        assert restored.recover_orders([]) == []
        assert restored.state['pending']['buy_allocations'] == {'688092.SH': 200}
        assert not broker.submissions
    finally:
        client.stop()


def test_bridge_disappearing_before_enqueue_preserves_retryable_journal(bridge, monkeypatch):
    from trading.execution_journal import ExecutionJournal
    from trading.executor import RebalanceExecutor
    _, broker, folder = bridge
    client = BridgeClient('test-account', folder)
    client.instance = 'old-instance'
    client.live_enabled = True
    monkeypatch.setattr(client, '_heartbeat', lambda: dict(instance='new-instance', live_enabled=True))
    day = date.today()
    journal = ExecutionJournal(folder / 'journal', 'test-account')
    journal.begin(dict(signal_date=day, trade_date=day, sell_orders=[],
                       buy_allocations={'688092.SH': 200}, buy_n_stocks=['688092.SH'],
                       prices={'688092.SH': 20.0}, limit_prices={}))
    def order(direction, code, shares, price, order_remark):
        return client.order_stock(None, code, direction, shares, 11, price, 'FlashMan', order_remark)
    executor = RebalanceExecutor(SimpleNamespace(order=order), journal=journal)
    with pytest.raises(BridgeUnavailable):
        executor.submit_order('BUY', '688092.SH', 200, 20.0, day, day)
    restored = ExecutionJournal(folder / 'journal', 'test-account')
    restored.load(day)
    assert restored.recover_orders([]) == []
    assert restored.state['requests'][0]['submission_state'] == 'not_submitted'
    assert not broker.submissions


def test_idle_health_check_drains_events_with_their_original_timestamps(bridge, monkeypatch):
    server, _, folder = bridge
    original = datetime(2026, 10, 30, 15, 1).timestamp()
    server.database.execute('INSERT INTO events(created,kind,payload) VALUES (?,?,?)',
                            (original, 'on_stock_trade', json.dumps({'order_id': 123})))
    server.database.commit()
    client = BridgeClient('test-account', folder)
    received = []
    client.register_callback(SimpleNamespace(on_stock_trade=received.append))
    monkeypatch.setattr(client, '_heartbeat', lambda: dict(instance='instance-a', live_enabled=True))
    client.start();client.connect()
    try:
        client.ensure_alive()
        assert len(received) == 1 and received[0]._bridge_event_time == original
        client.ensure_alive()
        assert len(received) == 1
    finally:client.stop()


@pytest.mark.parametrize('bad_cursor', [None, {}, -1, 1.2, True, 'broken'])
def test_corrupt_audit_cursor_replays_without_blocking_trading(bridge, bad_cursor):
    _, _, folder = bridge
    (folder / 'events_cursor.json').write_text(json.dumps({'seq': bad_cursor}), encoding='utf-8')
    client = BridgeClient('test-account', folder)
    callback = SimpleNamespace(on_stock_trade=lambda event: None)
    client.register_callback(callback)
    assert client.callback is callback and client.event_cursor == 0
