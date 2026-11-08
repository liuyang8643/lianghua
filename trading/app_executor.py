"""Autonomous broker-app execution. Python 3.6, no strategy imports.

The local planner can exit after publishing its JSON. This process owns all
account queries, holding differences, orders, cancellations and recovery.
"""
import datetime
import json
import logging
import os
from pathlib import Path
import queue
import time

from trading.account_snapshot import validate_account_snapshot
from trading.bridge_wire import BrokerConnectionLost, ConfirmedNotSubmitted, atomic_json, encode, open_database
from trading.execution_engine import RebalanceExecutor
from trading.execution_journal import ExecutionJournal
from trading.qmt_bridge import BrokerBridge, process_image
from trading.target_protocol import FILE_EXECUTOR_PROTOCOL, broker_fact_date, read_document, targets_to_pending, validate_target


def potentially_current_order(order, today):
    value = order.get('order_time') if isinstance(order, dict) else getattr(order, 'order_time', None)
    try:
        return broker_fact_date(value) >= today
    except ValueError:
        return True  # Unknown dates never permit bypassing an unowned order.


class AppBroker(BrokerBridge):
    def _guard_trade(self, payload, pid):
        super()._guard_trade(payload, pid)
        code = payload.get('stock_code')
        if code is not None:
            today = datetime.datetime.fromtimestamp(self.clock()).date()
            if any(order.stock_code == code and potentially_current_order(order, today)
                   and not self._owned_order(order)
                   for order in self._orders()):
                raise RuntimeError('Unowned order appeared on managed stock; reconcile before submission')
            # The additional query can block; recheck time/halt/ownership.
            super()._guard_trade(payload, pid)

    def _owner_check(self, pid):
        if pid != os.getpid():
            raise RuntimeError('Only this locked client executor may submit orders')
        actual = process_image(os.getppid())
        normalize = lambda path: os.path.normcase(os.path.realpath(path))
        if normalize(actual) not in [normalize(p) for p in self.config['client_images']]:
            raise RuntimeError('Executor no longer belongs to the brokerage client')
        if self.config.get('execution_mode') != 'file':
            raise RuntimeError('Autonomous file mode is not configured')


class AppTrader:
    def __init__(self, bridge):
        self.bridge = bridge
        self.account = bridge.account
        self.strategy_name = bridge.config['strategy_name']

    def query_account_snapshot(self):
        client = self.bridge.client
        try:
            asset = client.query_stock_asset(self.account)
            positions = client.query_stock_positions(self.account)
        except Exception as error:
            raise BrokerConnectionLost('Broker account snapshot unavailable: ' + str(error)) from error
        if asset is None or positions is None:
            raise BrokerConnectionLost('Broker account query returned no response')
        # Two valid responses may briefly have different valuation timestamps.
        # Wait for a coherent snapshot without tearing down a working SDK session.
        return validate_account_snapshot(asset, positions)

    def query_asset(self):
        return self.query_account_snapshot()[0]

    def query_orders(self):
        try:
            return self.bridge._orders()
        except Exception as error:
            raise BrokerConnectionLost('Broker order query unavailable: ' + str(error)) from error

    def query_order(self, order_id):
        return self.bridge.client.query_stock_order(self.account, order_id)

    def query_stock_position(self, code):
        return next((p for p in self.query_account_snapshot()[1] if p.stock_code == code), None)

    def query_not_submitted_intents(self, requests):
        result = self.bridge.dispatch('unsubmitted_intents',
                    {'account_id': self.bridge.config['account'], 'requests': requests}, os.getpid())
        return result

    def order(self, direction, code, shares, price, order_remark):
        cfg = self.bridge.config
        result = self.bridge._order(dict(account_id=cfg['account'],
            trade_date=datetime.datetime.fromtimestamp(self.bridge.clock()).date().isoformat(), stock_code=code,
            order_type=direction, volume=shares, price_type=cfg['price_types'][price is None],
            price=0.0 if price is None else price,
            strategy_name=self.strategy_name, order_remark=order_remark), os.getpid())
        return result

    def cancel_order(self, order_id):
        result = self.bridge.dispatch('cancel', dict(account_id=self.bridge.config['account'],
            trade_date=datetime.datetime.fromtimestamp(self.bridge.clock()).date().isoformat(),
            order_id=order_id), os.getpid())
        if result != 0:
            raise RuntimeError('Broker cancellation failed: ' + str(result))
        return result

    def query_trades(self):
        result = self.bridge.client.query_stock_trades(self.account)
        if result is None:
            raise RuntimeError('Broker trade query failed')
        return result


class ClientExecutionEngine(RebalanceExecutor):
    def __init__(self, trader, journal, directory):
        super().__init__(trader, journal=journal)
        self.directory = directory
        self.tick_callback = None

    def _on_tick(self):
        if self.tick_callback is not None:
            self.tick_callback()

    def _record_event(self, event, **fields):
        payload = dict(event=event, received_at=time.time(), **fields)
        for key, value in list(payload.items()):
            if isinstance(value, (datetime.date, datetime.datetime)):
                payload[key] = value.isoformat()
        database = self.trader.bridge.database
        database.execute('INSERT INTO events(created,kind,payload) VALUES (?,?,?)',
                         (time.time(), event, json.dumps(payload, allow_nan=False)))
        database.commit()


class FileExecutionService:
    def __init__(self, trader, directory, events=None, *, now_fn=None):
        self.trader = trader
        self.directory = Path(directory)
        self.journal = ExecutionJournal(self.directory / 'journal', trader.bridge.config['account'])
        self.engine = ClientExecutionEngine(trader, self.journal, self.directory)
        self.last_snapshot = 0
        self.last_transfers = 0
        self.events = events or queue.Queue()
        self.engine.tick_callback = self.maintenance
        self.now_fn = now_fn or datetime.datetime.now

    def maintenance(self):
        if (self.directory / 'stop.request').exists():
            raise SystemExit('Client strategy stop requested; broker app remains open')
        self.flush_events()
        if time.time() - self.last_snapshot >= 10:
            self.export_account(self.now_fn())
        self.persist_fact(self.directory / 'status.json', dict(updated=time.time(), pid=os.getpid(),
            parent_pid=os.getppid(), stage='executing',
            live_enabled=self.trader.bridge.config['live_enabled']))

    def flush_events(self):
        while not self.events.empty():
            kind, value, received = self.events.get_nowait()
            try:
                database = self.trader.bridge.database
                database.execute('INSERT INTO events(created,kind,payload) VALUES (?,?,?)',
                                 (received, kind, json.dumps(encode(value), allow_nan=False)))
                database.commit()
            except Exception as error:
                self.engine.logger.error('Broker audit write failed: %s', error)
                self.events.put((kind, value, received))
                break

    def persist_fact(self, path, document):
        try:
            atomic_json(path, document)
        except (OSError, TypeError, ValueError) as error:
            self.engine.logger.error('Account/audit export failed; execution continues: %s', error)

    def export_daily_record(self, snapshot):
        try:
            from trading.daily_records import write_executor_record
            write_executor_record(self.directory, self.journal.state, snapshot,
                                  self.trader.bridge.config.get('simulation', False))
        except Exception as error:
            self.engine.logger.error('Execution daily JSON failed; trading continues: %s', error)

    def export_account(self, now):
        asset, positions = self.trader.query_account_snapshot()
        orders = self.trader.query_orders()
        trades, trades_available = None, False
        try:
            trades = encode(self.trader.query_trades())
            trades_available = True
        except Exception as error:
            self.engine.logger.error('Trade audit query failed: %s', error)
        snapshot = dict(version=1, account=self.trader.bridge.config['account'],
                        executor_protocol=FILE_EXECUTOR_PROTOCOL,
                        captured_at=now.timestamp(), trade_date=now.date().isoformat(),
                        asset=encode(asset), positions=encode(positions), orders=encode(orders),
                        trades=trades, trades_available=trades_available)
        self.persist_fact(self.directory / ('account_' + now.date().isoformat() + '.json'), snapshot)
        self.persist_fact(self.directory / 'account_latest.json', snapshot)
        if now.time() >= datetime.time(15, 35):
            eod_path = self.directory / ('account_eod_' + now.date().isoformat() + '.json')
            if not eod_path.exists():
                self.persist_fact(eod_path, snapshot)
            if time.time() - self.last_transfers >= 300:
                try:
                    begin = (now.date() - datetime.timedelta(days=5)).strftime('%Y%m%d')
                    transfers = self.trader.bridge.client.query_bank_transfer_stream(
                        self.trader.account, begin, now.strftime('%Y%m%d'))
                    if transfers is None:
                        raise RuntimeError('Bank-transfer response missing')
                    self.persist_fact(self.directory / ('cash_flows_' + now.date().isoformat() + '.json'),
                                      dict(account=self.trader.bridge.config['account'],
                                           captured_at=now.timestamp(), rows=encode(transfers)))
                except Exception as error:
                    self.engine.logger.error('Cash-flow audit failed: %s', error)
                self.last_transfers = time.time()
        self.last_snapshot = time.time()
        return snapshot

    def tick(self, now=None):
        now = now or self.now_fn()
        self.flush_events()
        snapshot = self.export_account(now)
        cfg = self.trader.bridge.config
        today = now.date()
        target_path = self.directory / ('targets_' + today.isoformat() + '.json')
        saved = self.journal.load(today)
        if saved is not None:
            # The accepted document is frozen before any order. Local planner
            # exit, file expiry/removal or replacement cannot change recovery.
            document = self.journal.state['target_document']
            digest = self.journal.state['target_sha256']
        else:
            if not target_path.exists():
                return {'stage': 'waiting_for_local_target'}
            document, digest = read_document(target_path)
            validate_target(document, cfg['account'], today, now.timestamp(),
                            allow_simulation=cfg.get('simulation',False))
        positions = {p['stock_code']: p['volume'] for p in snapshot['positions']}
        pending = saved if saved is not None else targets_to_pending(document, positions)
        if not cfg['live_enabled']:
            result = dict(stage='preview', target_sha256=digest, pending=encode_pending(pending),
                          account=snapshot, actual_order_or_cancel_calls=0)
            atomic_json(self.directory / 'preview.json', result)
            return result
        if today.isoformat() not in cfg['trading_dates']:
            raise RuntimeError('Not a verified trading date')
        if not datetime.time(9, 25, 10) <= now.time() < datetime.time(15):
            if saved is not None and now.time() >= datetime.time(15, 35):
                self.export_daily_record(snapshot)
            return {'stage': 'outside_execution_window'}
        if saved is None:
            accepted_at = datetime.datetime.fromtimestamp(self.trader.bridge.clock())
            if accepted_at.date() != today or not datetime.time(9, 25, 10) <= accepted_at.time() < datetime.time(10):
                return {'stage': 'missed_initial_window'}
            # Any pre-existing unowned activity on managed symbols requires
            # explicit reconciliation before the first frozen difference.
            managed = set(document['targets'])
            if any(o['stock_code'] in managed and potentially_current_order(o, today)
                   for o in snapshot['orders']):
                raise RuntimeError('Existing broker orders require reconciliation before first file execution')
            validate_target(document, cfg['account'], today, self.trader.bridge.clock(),
                            allow_simulation=cfg.get('simulation',False))
            # Reserve immutable target hash before creating any order intent.
            self.journal.begin(pending, metadata={'target_sha256': digest,
                                                'target_document': document,
                                                'initial_account': {
                                                    'captured_at': snapshot['captured_at'],
                                                    'asset': snapshot['asset'],
                                                    'positions': [{'stock_code': p['stock_code'], 'volume': p['volume'],
                                                                   'can_use_volume': p['can_use_volume']}
                                                                  for p in snapshot['positions']]}})
        else:
            pending = saved
        if self.journal.state['completed']:
            covered = all(positions.get(code, 0) == row['shares']
                          for code, row in document['targets'].items())
            if covered:
                self.export_daily_record(snapshot)
            return {'stage': 'completed' if covered else 'external_position_change',
                    'target_sha256': digest}
        self.engine.execute(pending)
        final_snapshot = self.export_account(self.now_fn())
        final_positions = {p['stock_code']: p['volume'] for p in final_snapshot['positions']}
        covered = all(final_positions.get(code, 0) == row['shares']
                      for code, row in document['targets'].items())
        self.journal.state['completed'] = self.journal.state['completed'] and covered
        self.journal.save()
        if self.journal.state['completed']:
            self.export_daily_record(final_snapshot)
        return {'stage': 'completed' if self.journal.state['completed'] else 'pending',
                'target_sha256': digest}


def encode_pending(pending):
    result = dict(pending)
    for key in ('signal_date', 'trade_date'):
        result[key] = result[key].isoformat()
    return result


def connect_client(client, account, directory):
    """Reconnect the one SDK instance; never allocate another native session."""
    while not (directory / 'stop.request').exists():
        try:
            if client.connect() != 0:
                raise BrokerConnectionLost('Client execution connection failed')
            if client.subscribe(account) != 0:
                raise BrokerConnectionLost('Client account subscription failed')
            return
        except Exception as error:
            atomic_json(directory / 'status.json', dict(updated=time.time(), pid=os.getpid(),
                parent_pid=os.getppid(), stage='reconnecting', error=str(error)))
            print('WBR_FILE_RECONNECT', str(error), flush=True)
        for _ in range(5):
            if (directory / 'stop.request').exists():
                break
            time.sleep(1)
    raise SystemExit('Client strategy stop requested; broker app remains open')


def main():
    if (Path(__file__).resolve().parents[1] / 'data/live_file/stop.request').exists():
        raise SystemExit('Client strategy stop requested; broker app remains open')
    import msvcrt
    import psutil
    from contextlib import ExitStack
    root = Path(__file__).resolve().parents[1]
    directory = root / 'data/live_file'
    directory.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    if cfg.get('execution_mode') != 'file':
        raise RuntimeError('Client file execution mode must be explicitly configured')
    if cfg.get('simulation',False):
        raise RuntimeError('Production executor cannot use a simulation configuration')
    normalize = lambda path: os.path.normcase(os.path.realpath(path))
    if (normalize(cfg['root']) != normalize(root)
            or normalize(cfg['individual_config']) != normalize(root / 'configs/config.json')):
        raise RuntimeError('File executor configuration belongs to another repository')
    if normalize(process_image(os.getppid())) not in [normalize(p) for p in cfg['client_images']]:
        raise RuntimeError('Start this strategy inside the logged-in brokerage app')
    from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
    from xtquant.xttype import StockAccount
    os.chdir(str(root))
    logging.basicConfig(level=logging.INFO)
    with ExitStack() as cleanup:
        lock_path = root / 'data/live_trades/trading_main.lock'
        lock = cleanup.enter_context(lock_path.open('a+b'))
        lock.seek(0)
        lock.write(b'1')
        lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        process = psutil.Process(os.getpid())
        parent = psutil.Process(os.getppid())
        atomic_json(directory / 'identity.json', dict(pid=process.pid, created=process.create_time(),
            parent_pid=parent.pid, parent_created=parent.create_time(), cmdline=process.cmdline(),
            root=str(root), config=str(directory / 'config.json'), account=cfg['account']))
        database = open_database(directory / 'execution.sqlite3')
        cleanup.callback(database.close)
        services = []
        cleanup.callback(lambda: services and services[0].flush_events())
        client = XtQuantTrader(cfg['userdata'], int(time.time() * 1000) % 2000000000)
        client.set_timeout(5)
        events = queue.Queue()
        class Callback(XtQuantTraderCallback):
            def on_stock_order(self, value):
                events.put(('on_stock_order', value, time.time()))

            def on_stock_trade(self, value):
                events.put(('on_stock_trade', value, time.time()))

            def on_order_error(self, value):
                events.put(('on_order_error', value, time.time()))

            def on_cancel_error(self, value):
                events.put(('on_cancel_error', value, time.time()))
        if cfg['live_enabled']:
            client.register_callback(Callback())
        client.start()
        cleanup.callback(client.stop)
        account = StockAccount(cfg['account'], 'STOCK')
        connect_client(client, account, directory)
        bridge = AppBroker(cfg, client, account, database)
        service = FileExecutionService(AppTrader(bridge), directory, events)
        services.append(service)
        print('WBR_FILE_EXECUTOR_READY', os.getpid(), 'live=' + str(cfg['live_enabled']), flush=True)
        while not (directory / 'stop.request').exists():
            try:
                result = service.tick()
                atomic_json(directory / 'status.json', dict(updated=time.time(), pid=os.getpid(),
                            parent_pid=os.getppid(), live_enabled=cfg['live_enabled'], **result))
            except BrokerConnectionLost as error:
                atomic_json(directory / 'status.json', dict(updated=time.time(), pid=os.getpid(),
                    parent_pid=os.getppid(), stage='reconnecting', error=str(error)))
                print('WBR_FILE_CONNECTION_LOST', str(error), flush=True)
                service.flush_events()
                connect_client(client, account, directory)
                service = FileExecutionService(AppTrader(bridge), directory, events)
                services[0] = service
            except Exception as error:
                atomic_json(directory / 'status.json', dict(updated=time.time(), pid=os.getpid(),
                            stage='blocked', error=type(error).__name__ + ': ' + str(error)))
                print('WBR_FILE_EXECUTOR_BLOCKED', str(error), flush=True)
            time.sleep(1)


if __name__ == '__main__':
    main()
