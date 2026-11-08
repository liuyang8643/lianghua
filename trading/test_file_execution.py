from datetime import date, datetime
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from trading.app_executor import AppTrader, FileExecutionService
from trading.bridge_wire import ConfirmedNotSubmitted, atomic_json
from trading.execution_engine import OrderMonitor
from trading.target_protocol import targets_to_pending
from trading.test_bridge import bridge


def test_incoherent_valuation_does_not_masquerade_as_sdk_disconnect(bridge):
    from trading.bridge_wire import BrokerConnectionLost
    server, broker, root = bridge
    prepare(server, broker, root)
    broker.query_stock_asset = lambda account: NS(cash=1000, total_asset=3000, market_value=2000)
    trader = AppTrader(server)
    with pytest.raises(ValueError, match='market value'):
        trader.query_account_snapshot()
    broker.query_stock_asset = lambda account: None
    with pytest.raises(BrokerConnectionLost):
        trader.query_account_snapshot()


def prepare(server, broker, root, *, live=True):
    server.clock = lambda: datetime(2026, 11, 2, 9, 30).timestamp()
    server.config['live_enabled'] = live
    server.config['execution_mode'] = 'file'
    directory = root / 'file'
    directory.mkdir()
    broker.positions = [NS(stock_code='600001.SH', volume=100, can_use_volume=100, market_value=1000)]
    broker.query_stock_positions = lambda account: broker.positions
    broker.query_stock_trades = lambda account: []
    broker.query_stock_order = lambda account, oid: next((o for o in broker.orders if o.order_id == oid), None)
    target = dict(version=1, id='abcdefabcdef', account='test-account', trade_date='2026-11-02',
                  executor_protocol='wbr-file-targets-v2-capacity',
                  strategy_id='test',model_sha256='0'*64,max_positions=40,
                  generated_at=server.clock() - 10, expires=server.clock() + 30,
                  targets={'600001.SH': {'shares': 200, 'open_price': 10, 'freeze_price': 11}},
                  buy_priority=['600001.SH'])
    atomic_json(directory / 'targets_2026-11-02.json', target)
    return directory, target


def fill_broker(broker):
    def submit(account, code, side, qty, price_type, price, strategy, remark):
        broker.submissions.append((code, side, qty, remark))
        oid = len(broker.submissions)
        broker.orders.append(NS(order_id=oid, stock_code=code, order_type=side,
            order_volume=qty, traded_volume=qty, order_status=56,
            strategy_name=strategy, order_remark=remark, traded_price=10, status_msg=''))
        broker.positions[0].volume += qty if side == 23 else -qty
        return oid
    broker.order_stock = submit


@pytest.fixture
def single_cycle(monkeypatch):
    monkeypatch.setattr(OrderMonitor, '_can_submit_now', lambda *args: True)
    monkeypatch.setattr(OrderMonitor, '_main_loop', lambda self: self._submit_affordable_buys())


def test_client_executes_file_without_local_rpc_and_repeated_reads_do_not_buy_again(bridge, single_cycle):
    server, broker, root = bridge
    directory, _ = prepare(server, broker, root)
    fill_broker(broker)
    service = FileExecutionService(AppTrader(server), directory)
    service.engine.SETTLE_WAIT_SEC = 0
    result = service.tick(datetime(2026, 11, 2, 9, 30))
    assert result['stage'] == 'completed'
    assert broker.submissions[0][1:3] == (23, 100)
    service.tick(datetime(2026, 11, 2, 9, 30))
    assert len(broker.submissions) == 1
    assert server.database.execute('SELECT count(*) FROM requests').fetchone()[0] == 0


def test_client_recovers_after_target_file_deleted_and_expired(bridge, single_cycle):
    server, broker, root = bridge
    directory, _ = prepare(server, broker, root)
    fill_broker(broker)
    first = FileExecutionService(AppTrader(server), directory)
    first.engine.SETTLE_WAIT_SEC = 0
    first.tick(datetime(2026, 11, 2, 9, 30))
    (directory / 'targets_2026-11-02.json').unlink()
    second = FileExecutionService(AppTrader(server), directory)
    assert second.tick(datetime(2026, 11, 2, 10, 2))['stage'] == 'completed'
    assert len(broker.submissions) == 1


def test_readonly_client_exports_positions_and_diff_without_journal_or_orders(bridge):
    server, broker, root = bridge
    directory, _ = prepare(server, broker, root, live=False)
    result = FileExecutionService(AppTrader(server), directory).tick(datetime(2026, 11, 2, 9, 30))
    assert result['stage'] == 'preview'
    assert result['pending']['buy_allocations'] == {'600001.SH': 100}
    assert broker.submissions == []
    assert not (directory / 'journal/execution_2026-11-02.json').exists()


def test_sdk_negative_order_result_is_not_a_successful_submission(bridge):
    from trading.qmt_bridge import UncertainOrder
    server, broker, root = bridge
    prepare(server, broker, root)
    broker.order_stock = lambda *args: -1
    with pytest.raises(UncertainOrder):
        AppTrader(server).order(23, '600001.SH', 100, 10, 'wbr:abcdefabcdef:000000')
    assert server.database.execute('SELECT state FROM order_intents').fetchone()[0] == 'executing'


def test_first_sell_is_limited_by_real_sellable_shares(bridge, single_cycle):
    server, broker, root = bridge
    directory, target = prepare(server, broker, root)
    target['targets']['600001.SH']['shares'] = 0
    atomic_json(directory / 'targets_2026-11-02.json', target)
    broker.positions[0].can_use_volume = 0
    fill_broker(broker)
    service = FileExecutionService(AppTrader(server), directory)
    service.engine.SETTLE_WAIT_SEC = 0
    assert service.tick(datetime(2026, 11, 2, 9, 30))['stage'] == 'pending'
    assert broker.submissions == []


def test_account_export_write_failure_does_not_interrupt_started_execution(bridge, monkeypatch):
    server, broker, root = bridge
    directory, _ = prepare(server, broker, root, live=False)
    import trading.app_executor as module
    def fail(*args):
        raise PermissionError('Export unavailable')
    monkeypatch.setattr(module, 'atomic_json', fail)
    service = FileExecutionService(AppTrader(server), directory)
    assert service.export_account(datetime(2026, 11, 2, 9, 30))['positions'][0]['volume'] == 100


def test_unlisted_holdings_are_never_liquidated():
    target = {'trade_date': '2026-11-02', 'max_positions':40, 'targets': {
        '600001.SH': {'shares': 200, 'open_price': 10, 'freeze_price': 11}},
        'buy_priority': ['600001.SH']}
    pending = targets_to_pending(target, {'600001.SH': 100, '600002.SH': 10000})
    assert pending['sell_orders'] == []
    assert pending['buy_allocations'] == {'600001.SH': 100}


def test_local_planner_imports_and_dependencies_never_load_qmt_in_fresh_process():
    import subprocess
    import sys
    code = '''
import importlib.abc,sys
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname == 'xtquant' or fullname.startswith('xtquant.'):
   raise AssertionError('QMT import prohibited: ' + fullname)
sys.meta_path.insert(0,Deny())
import trading.plan_publisher, trading.planner_watchdog
import strategy_release.runtime, env.planner, offline_data
import data.update_live, data.kline_mootdx, data.build_runtime
print('NO_QMT_PASS')
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'NO_QMT_PASS' in result.stdout


def test_foreign_order_appearing_immediately_before_sdk_blocks_new_submission(bridge):
    from trading.app_executor import AppBroker
    server, broker, root = bridge
    prepare(server, broker, root)
    app = AppBroker(server.config, broker, server.account, server.database,
                    clock=server.clock, owner_check=lambda pid: None)
    calls = []
    def orders(account, cancelable_only=False):
        calls.append(1)
        if len(calls) < 3:
            return []
        return [NS(stock_code='600001.SH', strategy_name='', order_remark='', order_id=0)]
    broker.query_stock_orders = orders
    with pytest.raises(ConfirmedNotSubmitted, match='guard rejected'):
        AppTrader(app).order(23, '600001.SH', 100, 10, 'wbr:abcdefabcdef:000000')
    assert broker.submissions == []
    assert app.database.execute('SELECT state FROM order_intents').fetchone()[0] == 'not_submitted'


def test_failed_cancellation_does_not_count_as_success(bridge):
    server, broker, root = bridge
    prepare(server, broker, root)
    trader = AppTrader(server)
    trader.order(23, '600001.SH', 100, 10, 'wbr:abcdefabcdef:000000')
    broker.orders = [NS(order_id=123, stock_code='600001.SH', strategy_name='FlashMan',
                       order_remark='wbr:abcdefabcdef:000000', order_type=23,
                       order_volume=100, traded_volume=0, order_status=50)]
    broker.cancel_order_stock = lambda *args: -1
    with pytest.raises(RuntimeError, match='cancellation failed'):
        trader.cancel_order(123)


def test_partial_fill_recovery_after_source_removed_never_submits_twice(bridge, single_cycle):
    server, broker, root = bridge
    directory, _ = prepare(server, broker, root)
    fill_broker(broker)
    fill = broker.order_stock
    def partial(*args):
        result = fill(*args)
        broker.orders[0].traded_volume = 50
        broker.orders[0].order_status = 55
        broker.positions[0].volume = 150
        return result
    broker.order_stock = partial
    first = FileExecutionService(AppTrader(server), directory)
    first.engine.SETTLE_WAIT_SEC = 0
    assert first.tick(datetime(2026, 11, 2, 9, 30))['stage'] == 'pending'
    (directory / 'targets_2026-11-02.json').unlink()
    broker.orders[0].traded_volume = 100
    broker.orders[0].order_status = 56
    broker.positions[0].volume = 200
    second = FileExecutionService(AppTrader(server), directory)
    second.engine.SETTLE_WAIT_SEC = 0
    assert second.tick(datetime(2026, 11, 2, 10, 2))['stage'] == 'completed'
    assert len(broker.submissions) == 1


def test_controlled_stop_interrupts_execution_without_touching_client(bridge):
    server, broker, root = bridge
    directory, _ = prepare(server, broker, root)
    service = FileExecutionService(AppTrader(server), directory)
    (directory / 'stop.request').write_text('stop')
    with pytest.raises(SystemExit, match='broker app remains open'):
        service.engine._on_tick()
    assert broker.submissions == []


def test_competing_publication_cannot_overwrite_target(tmp_path, monkeypatch):
    from trading import target_protocol as protocol
    path = tmp_path / 'target.json'
    real_link = protocol.os.link
    competing = {'id': 'other', 'targets': {}}
    def race(source, destination):
        atomic_json(path, competing)
        real_link(source, destination)
    monkeypatch.setattr(protocol.os, 'link', race)
    with pytest.raises(RuntimeError, match='Concurrent publisher'):
        protocol.write_immutable_target(path, {'id': 'first', 'targets': {}})
    assert json.loads(path.read_text()) == competing
    assert not list(tmp_path.glob('*.tmp'))


def test_file_bootstrap_does_not_reallocate_sdk_after_fatal_start_failure(tmp_path):
    import subprocess
    import sys
    from trading.bridge_setup import bootstrap_source
    source = tmp_path / 'trading/app_executor.py'
    source.parent.mkdir()
    source.write_text('''import os,time
from pathlib import Path
time.sleep=lambda seconds: None
p=Path(__file__).with_suffix('.attempts')
rows=p.read_text().splitlines() if p.exists() else []
rows.append(str(os.getpid()))
p.write_text('\\n'.join(rows))
if len(rows)==1: raise RuntimeError('simulated SDK connection lost')
''')
    entry = tmp_path / 'entry.py'
    entry.write_text(bootstrap_source(tmp_path, 'trading/app_executor.py'))
    result = subprocess.run([sys.executable, str(entry)], capture_output=True, text=True)
    assert result.returncode != 0
    attempts = source.with_suffix('.attempts').read_text().splitlines()
    assert len(attempts) == 1


def test_client_reconnect_reuses_instance_and_obeys_stop(tmp_path, monkeypatch):
    import trading.app_executor as module
    client = NS(connect_calls=0, subscribe_calls=0)
    def connect():
        client.connect_calls += 1
        return -1 if client.connect_calls < 3 else 0
    def subscribe(account):
        client.subscribe_calls += 1
        return 0
    client.connect, client.subscribe = connect, subscribe
    monkeypatch.setattr(module.time, 'sleep', lambda seconds: None)
    module.connect_client(client, 'account', tmp_path)
    module.connect_client(client, 'account', tmp_path)
    assert (client.connect_calls, client.subscribe_calls) == (4, 2)
    (tmp_path / 'stop.request').touch()
    with pytest.raises(SystemExit):
        module.connect_client(client, 'account', tmp_path)
    assert client.connect_calls == 4


@pytest.mark.parametrize('start_fails', [False, True])
def test_client_main_owns_one_sdk_lifetime(tmp_path, monkeypatch, start_fails):
    import sys
    from types import ModuleType
    import trading.app_executor as module
    root = tmp_path
    directory = root / 'data/live_file'
    directory.mkdir(parents=True)
    (root / 'data/live_trades').mkdir()
    config = dict(root=str(root), individual_config=str(root / 'configs/config.json'),
                  account='SIMULATED', execution_mode='file', live_enabled=False,
                  client_images=['FAKE_PARENT'], userdata='FAKE', strategy_name='test')
    atomic_json(directory / 'config.json', config)
    counts = dict(created=0, started=0, stopped=0, connected=0, services=0, ticks=0)
    class SDK:
        def __init__(self, *args):
            counts['created'] += 1
        def set_timeout(self, value):
            pass
        def start(self):
            counts['started'] += 1
            if start_fails:
                raise RuntimeError('native start failed')
        def stop(self):
            counts['stopped'] += 1
        def connect(self):
            counts['connected'] += 1
            return 0
        def subscribe(self, account):
            return 0
    class Service:
        def __init__(self, trader, path, events):
            assert isinstance(trader.bridge.client, SDK)
            assert path == directory
            counts['services'] += 1
        def flush_events(self):
            pass
        def tick(self):
            counts['ticks'] += 1
            if counts['ticks'] == 1:
                raise module.BrokerConnectionLost('fake disconnect')
            (directory / 'stop.request').touch()
            return dict(stage='completed')
    package, trader_module, types_module = [ModuleType(n) for n in
                                           ('xtquant', 'xtquant.xttrader', 'xtquant.xttype')]
    trader_module.XtQuantTrader, trader_module.XtQuantTraderCallback = SDK, object
    types_module.StockAccount = lambda *args: 'SIMULATED'
    for name, fake in [('xtquant', package), ('xtquant.xttrader', trader_module),
                       ('xtquant.xttype', types_module)]:
        monkeypatch.setitem(sys.modules, name, fake)
    monkeypatch.setattr(module, '__file__', str(root / 'trading/app_executor.py'))
    monkeypatch.setattr(module, 'process_image', lambda pid: 'FAKE_PARENT')
    monkeypatch.setattr(module, 'FileExecutionService', Service)
    monkeypatch.setattr(module.time, 'sleep', lambda seconds: None)
    monkeypatch.chdir(root)
    if start_fails:
        with pytest.raises(RuntimeError, match='native start failed'):
            module.main()
        assert counts == dict(created=1, started=1, stopped=0, connected=0, services=0, ticks=0)
    else:
        module.main()
        assert counts == dict(created=1, started=1, stopped=1, connected=2, services=2, ticks=2)


def test_no_qmt_fallback_when_mootdx_fails(monkeypatch):
    import data.update_kline as module
    import data.update_tencent_close as closing
    def fail(*args, **kwargs):
        raise RuntimeError('All mootdx nodes failed')
    def forbidden(*args, **kwargs):
        raise AssertionError('QMT fallback must not run')
    monkeypatch.setattr(module.mdx, 'update_recent', fail)
    monkeypatch.setattr(module, '_qmt_update', forbidden)
    calls = []
    monkeypatch.setattr(closing, 'repair_session', lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(module.mdx, 'resolve_recent_range', lambda *a: ('20261102', '20261102', None))
    module.update_recent(1, allow_qmt=False)
    assert calls == [((date(2026, 11, 2), None), {'collect': False})]
    with pytest.raises(RuntimeError, match='prohibits QMT'):
        module.update_recent(3, allow_qmt=False)


def test_production_main_cli_imports_no_broker_sdk():
    import subprocess
    import sys
    code = ("import sys,runpy; from trading.planner_watchdog import install_no_qmt_guard; "
            "install_no_qmt_guard(); sys.argv=['trading.main','--help']; "
            "runpy.run_module('trading.main',run_name='__main__')")
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert '--directory' in result.stdout


def test_installer_preserves_registered_entry_backup_and_defaults_readonly(tmp_path):
    from trading.file_setup import prepare
    root = tmp_path
    old = root / 'data/live_file'
    old.mkdir(parents=True)
    broker_python = root / 'broker/bin.x64/pythonw.exe'
    entry = root / 'broker/mpython/WBR_EXEC_BRIDGE.py'
    entry.parent.mkdir(parents=True)
    entry.write_text('OLD_REGISTERED_ENTRY')
    atomic_json(old / 'config.json', dict(account='SIMULATED', broker_python=str(broker_python)))
    directory = prepare(root)
    cfg = json.loads((directory / 'config.json').read_text())
    assert cfg['execution_mode'] == 'file' and cfg['live_enabled'] is False
    assert (directory / 'WBR_EXEC_BRIDGE.before_file_mode.py').read_text() == 'OLD_REGISTERED_ENTRY'
    assert 'trading/app_executor.py' in entry.read_text()


def test_previous_baseline_rejects_unbound_or_other_account(tmp_path):
    from trading.plan_publisher import previous_baseline
    previous = date(2026, 11, 2)
    atomic_json(tmp_path / 'account_eod_2026-11-02.json',
                dict(account='wrong-account', trade_date='2026-11-02'))
    with pytest.raises(ValueError, match='account/date'):
        previous_baseline(tmp_path, previous, 'SIMULATED')


def test_local_reporting_dependencies_have_no_sdk_import_in_fresh_process():
    import subprocess
    import sys
    code = '''
import importlib.abc,sys
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname == 'xtquant' or fullname.startswith('xtquant.'):
   raise AssertionError('Forbidden SDK import: ' + fullname)
sys.meta_path.insert(0,Deny())
import trading.file_account,trading.file_control,trading.file_post_close
import trading.post_close,trading.persistence,trading.day_board,trading.report,trading.lark_service
print('FILE_REPORTING_NO_SDK_PASS')
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cached_previous_day_orders_do_not_block_today():
    from trading.app_executor import potentially_current_order
    prior = NS(order_time=datetime(2026, 11, 2, 9, 30).timestamp())
    assert not potentially_current_order(prior, date(2026, 11, 3))
    assert potentially_current_order(NS(), date(2026, 11, 3))


def test_first_execution_after_ten_is_rejected_even_if_file_not_expired(bridge):
    server, broker, root = bridge
    directory, target = prepare(server, broker, root)
    server.clock = lambda: datetime(2026, 11, 2, 13).timestamp()
    target['expires'] = datetime(2026, 11, 2, 15).timestamp()
    atomic_json(directory / 'targets_2026-11-02.json', target)
    service = FileExecutionService(AppTrader(server), directory)
    assert service.tick(datetime(2026, 11, 2, 13))['stage'] == 'missed_initial_window'
    assert not (directory / 'journal/execution_2026-11-02.json').exists()
    assert not broker.submissions


def test_first_execution_rechecks_clock_after_slow_account_query(bridge):
    server, broker, root = bridge
    directory, target = prepare(server, broker, root)
    target['expires'] = datetime(2026, 11, 2, 15).timestamp()
    atomic_json(directory / 'targets_2026-11-02.json', target)
    original = broker.query_stock_positions
    def slow_query(account):
        server.clock = lambda: datetime(2026, 11, 2, 10, 0, 5).timestamp()
        return original(account)
    broker.query_stock_positions = slow_query
    service = FileExecutionService(AppTrader(server), directory)
    assert service.tick(datetime(2026, 11, 2, 9, 59, 59))['stage'] == 'missed_initial_window'
    assert not broker.submissions


def test_account_file_reads_fresh_clock_after_read(monkeypatch):
    import trading.plan_publisher as module
    initial = datetime(2026, 11, 2, 9, 25, 10).timestamp()
    clock = [initial]
    snapshot = dict(version=1, account='SIMULATED', trade_date='2026-11-02', captured_at=initial + 2,
        asset=dict(cash=1000, total_asset=2000, market_value=1000),
        positions=[dict(stock_code='600001.SH', volume=100, can_use_volume=100, market_value=1000)])
    def read(path):
        clock[0] = initial + 3
        return snapshot, 'hash'
    monkeypatch.setattr(module, 'read_document', read)
    result, _, _ = module.load_account_file('unused', 'SIMULATED', date(2026, 11, 2), lambda: clock[0])
    assert result is snapshot


def test_corrupt_calendar_enters_repair_loop(tmp_path, monkeypatch):
    import trading.planner_watchdog as planner
    import utils.stock.time as calendar
    import data.update_all as updates
    def bad_calendar():
        raise ValueError('corrupt calendar')
    called = []
    monkeypatch.setattr(calendar, '_get_trading_calendar_state', bad_calendar)
    monkeypatch.setattr(updates, '_update_trading_calendar', lambda: called.append('repair'))
    def stop(seconds):
        raise SystemExit('test stop')
    monkeypatch.setattr(planner.time, 'sleep', stop)
    with pytest.raises(SystemExit, match='test stop'):
        planner.run('unused', tmp_path)
    assert called == ['repair']
    assert json.loads((tmp_path / 'planner_status.json').read_text())['stage'] == 'calendar_blocked'


def test_post_close_report_uses_one_fixed_account_snapshot(tmp_path):
    from trading.file_account import FileAccount
    path = tmp_path / 'account_2026-11-02.json'
    snapshot = dict(version=1, account='SIMULATED', trade_date='2026-11-02',
        captured_at=datetime(2026, 11, 2, 15, 35).timestamp(),
        asset=dict(cash=0, total_asset=1000, market_value=1000),
        positions=[dict(stock_code='600001.SH', volume=100, can_use_volume=100, market_value=1000)],
        trades_available=True, trades=[])
    atomic_json(path, snapshot)
    account = FileAccount('SIMULATED', tmp_path, trade_date=date(2026, 11, 2))
    assert account.query_asset().total_asset == 1000
    snapshot['asset'].update(total_asset=2000, market_value=2000)
    snapshot['positions'][0].update(volume=200, can_use_volume=200, market_value=2000)
    atomic_json(path, snapshot)
    assert account.query_positions()[0].volume == 100
    assert account.query_asset().total_asset == 1000


def test_publisher_requires_the_reloaded_executor_before_calculating(tmp_path,monkeypatch):
    from trading import plan_publisher,startup_data
    from utils.stock import time as calendar
    now=datetime(2026,11,9,9,25,10)
    atomic_json(tmp_path/'config.json',dict(account='test'))
    atomic_json(tmp_path/'account_latest.json',dict(version=1,account='test',trade_date='2026-11-09',
        captured_at=now.timestamp(),asset=dict(cash=10000,total_asset=10000,market_value=0),positions=[]))
    monkeypatch.setattr(calendar,'_get_trading_calendar_state',lambda:([date(2026,11,6),date(2026,11,9)],None,date(2026,12,31)))
    monkeypatch.setattr(calendar,'get_last_trading_day',lambda d:date(2026,11,6))
    monkeypatch.setattr(startup_data,'validate_startup_runtime',lambda *a:None)
    with pytest.raises(RuntimeError,match='Reload WBR_EXEC_BRIDGE'):
        plan_publisher.publish_target('unused',tmp_path,now=now)
    assert not (tmp_path/'targets_2026-11-09.json').exists()


def test_real_executor_rejects_simulation_configuration_before_sdk_import(tmp_path,monkeypatch):
    import builtins
    import trading.app_executor as app
    directory=tmp_path/'data/live_file';directory.mkdir(parents=True)
    atomic_json(directory/'config.json',dict(execution_mode='file',simulation=True))
    monkeypatch.setattr(app,'__file__',str(tmp_path/'trading/app_executor.py'))
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.startswith(('xtquant.xttrader','xtquant.xtdata')):
            pytest.fail('A simulation configuration imported a real brokerage SDK')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    with pytest.raises(RuntimeError,match='simulation configuration'):
        app.main()


def test_conflicting_duplicate_trade_is_not_silently_deduplicated(monkeypatch):
    from trading.file_account import FileAccount
    account = FileAccount('SIMULATED', trade_date=date(2026, 11, 2))
    trade = dict(stock_code='600001.SH', order_type=23, traded_id='real-fill-1',
                 traded_time=datetime(2026, 11, 2, 10).timestamp(),
                 traded_volume=100, traded_price=10, traded_amount=1000, order_id=1)
    snapshot = dict(trades_available=True, trades=[trade, dict(trade)],
        orders=[dict(stock_code='600001.SH', order_type=23, traded_volume=100,
                     order_time=trade['traded_time'])])
    monkeypatch.setattr(account, '_snapshot', lambda: snapshot)
    assert len(account.query_all_trades()) == 1
    snapshot['trades'][1].update(traded_price=20, traded_amount=2000)
    with pytest.raises(RuntimeError, match='Conflicting duplicate'):
        account.query_all_trades()


def test_mootdx_stock_list_validates_real_lists_without_requiring_daily_endpoint(tmp_path, monkeypatch):
    import pandas as pd
    import data.update_all as module
    import data.kline_mootdx as mdx
    frames = {0: pd.DataFrame({'code': ['000001', '301049', '159915']}),
              1: pd.DataFrame({'code': ['600000', '688092', '900901']})}
    class Quotes:
        def stocks(self, market):
            return frames[market]
        def stock_count(self, market):
            return len(frames[market])
    seen = []
    def connect(**kwargs):
        seen.append(kwargs)
        return Quotes()
    monkeypatch.setattr(mdx, '_connect_mootdx', connect)
    monkeypatch.setattr(mdx, '_close_mootdx', lambda quote: None)
    monkeypatch.setattr(module, 'DATA_DIR', tmp_path)
    module._update_stock_list(source='mootdx')
    result = pd.read_parquet(tmp_path / 'stock_list/stock_list.parquet')
    assert set(result['stock_code']) == {'000001.SZ', '301049.SZ', '600000.SH', '688092.SH'}
    assert seen == [{'require_daily': False}]
