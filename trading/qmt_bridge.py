# coding: utf-8
"""Client-side execution only. Python 3.6; never imports strategy/factor modules."""
import datetime
import json
import math
import os
from pathlib import Path
import queue
import re
import sys
import time
import uuid

from trading.bridge_wire import ConfirmedNotSubmitted, atomic_json, encode, open_database


class UncertainOrder(RuntimeError):
    pass


def process_image(pid):
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        raise RuntimeError('Request owner process is not readable/alive')
    try:
        exit_code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)) or exit_code.value != 259:
            raise RuntimeError('Request owner process has exited')
        size = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(size.value)
        if not kernel.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            raise RuntimeError('Cannot verify process image')
        return buffer.value
    finally:
        kernel.CloseHandle(handle)


class BrokerBridge:
    def __init__(self, config, client, account, database, *, clock=time.time,
                 owner_check=None, instance=None):
        self.config = config
        self.client = client
        self.account = account
        self.database = database
        self.clock = clock
        self.instance = instance or uuid.uuid4().hex
        self.owner_check = owner_check or self._owner_check
        self.request_deadline = None

    def _owner_check(self, pid):
        import msvcrt
        import psutil
        expected = Path(self.config['root']) / 'data/live_trades/trading_main.pid'
        if not expected.exists() or expected.read_text(encoding='utf-8').strip() != str(pid):
            raise RuntimeError('Only the locked live main process may request trades')
        image = process_image(pid)
        expected_python = os.path.normcase(os.path.realpath(self.config['project_python']))
        # Windows venv uses a redirector; its real child executable is recorded
        # explicitly at deployment rather than trusting an arbitrary python.exe.
        allowed = [expected_python] + [os.path.normcase(os.path.realpath(path))
                                      for path in self.config['project_python_images']]
        if os.path.normcase(os.path.realpath(image)) not in allowed:
            raise RuntimeError('Trade request owner uses another Python environment')
        process = psutil.Process(pid)
        identity_path = expected.with_name('trading_main.identity.json')
        identity = json.loads(identity_path.read_text(encoding='utf-8'))
        same_path = lambda a, b: os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))
        arguments = process.cmdline()
        if (identity['pid'] != pid
                or abs(process.create_time() - identity['created']) > 0.1
                or not same_path(process.cwd(), self.config['root'])
                or identity['cmdline'] != arguments
                or identity['backend'] != 'bridge'
                or not same_path(identity['config'], self.config['individual_config'])
                or not any(arguments[i:i + 2] == ['-m', 'trading.main']
                           for i in range(len(arguments) - 1))
                or '--dry-run' in arguments or '--skip' in arguments):
            raise RuntimeError('Live main process identity is stale or belongs to another repository')
        # Verify the process still holds the same Windows byte-range lock used
        # by filelock, rather than trusting a PID marker without a live lock.
        with expected.with_name('trading_main.lock').open('r+b') as stream:
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                if exc.errno != 13:
                    raise
            else:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                raise RuntimeError('Live main process lock is not held')

    def _guard_trade(self, payload, pid):
        if not self.config['live_enabled']:
            raise RuntimeError('Broker bridge is read-only')
        self.owner_check(pid)
        # Process inspection can itself block. Check time and halt after it,
        # immediately before the caller reaches the real broker API.
        if self.request_deadline is not None and self.clock() >= self.request_deadline:
            raise RuntimeError('Trade request expired before submission')
        if Path(self.config['root'], 'data/live_trades/trading_watchdog.halt').exists():
            raise RuntimeError('Emergency halt marker prohibits new trading actions')
        now = datetime.datetime.fromtimestamp(self.clock())
        if now.date().isoformat() not in self.config['trading_dates']:
            raise RuntimeError('Not a verified trading day')
        t = now.time()
        if not (datetime.time(9, 30) <= t < datetime.time(11, 30)
                or datetime.time(13) <= t < datetime.time(15)):
            raise RuntimeError('Broker bridge prohibits orders outside continuous trading')
        if payload['trade_date'] != now.date().isoformat():
            raise RuntimeError('Trade request date is stale')

    def _orders(self):
        orders = self.client.query_stock_orders(self.account, cancelable_only=False)
        if orders is None:
            raise RuntimeError('Broker order query failed; cannot establish order ownership')
        return orders

    def _owned_order(self, order):
        if (order.strategy_name != self.config['strategy_name']
                or not re.fullmatch(r'wbr:[a-f0-9]{12}:\d{6}', order.order_remark)):
            return False
        row = self.database.execute(
            'SELECT specification FROM order_intents WHERE remark=?', (order.order_remark,)).fetchone()
        if row is not None:
            intent = json.loads(row[0])
            return (intent['stock_code'] == order.stock_code
                    and intent['order_type'] == order.order_type
                    and intent['volume'] == order.order_volume)
        # A native-to-bridge cutover can recover a durable original execution
        # journal, but cannot cancel another repo's similarly named strategy.
        today = datetime.datetime.fromtimestamp(self.clock()).date().isoformat()
        path = Path(self.config['root'], 'data/live_trades/execution_' + today + '.json')
        if not path.exists():
            return False
        journal = json.loads(path.read_text(encoding='utf-8'))
        if journal['account'] != self.config['account']:
            return False
        return any(r['remark'] == order.order_remark and r['code'] == order.stock_code
                   and r['shares'] == order.order_volume
                   and (23 if r['direction'] == 'BUY' else 24) == order.order_type
                   for r in journal['requests'])

    def _confirmed_order_id(self, payload):
        matches = [o for o in self._orders() if o.order_remark == payload['order_remark']]
        if len(matches) != 1:
            raise UncertainOrder('Submission status unknown; automatic resend prohibited')
        order = matches[0]
        if (order.stock_code != payload['stock_code'] or order.order_volume != payload['volume']
                or order.order_type != payload['order_type']
                or order.strategy_name != payload['strategy_name'] or int(order.order_id) <= 0):
            raise UncertainOrder('Broker order does not match durable intent')
        return int(order.order_id)

    def _order(self, payload, pid):
        shares = payload['volume']
        code = payload['stock_code']
        remark = payload['order_remark']
        if (type(shares) is not int or shares <= 0
                or not re.fullmatch(r'\d{6}\.(SH|SZ|BJ)', code)
                or payload['order_type'] not in (23, 24)
                or payload['price_type'] not in self.config['price_types']
                or not math.isfinite(payload['price']) or payload['price'] < 0
                or (payload['price_type'] == self.config['price_types'][0] and payload['price'] <= 0)
                or payload['strategy_name'] != self.config['strategy_name']
                or not re.fullmatch(r'wbr:[a-f0-9]{12}:\d{6}', remark)):
            raise ValueError('Invalid execution intent')
        spec = json.dumps(payload, sort_keys=True, allow_nan=False)
        existing = self.database.execute(
            'SELECT specification,state,result FROM order_intents WHERE remark=?',
            (remark,)).fetchone()
        if existing:
            if existing[0] != spec:
                raise RuntimeError('Same order remark used for a different intent')
            if existing[1] == 'done' and existing[2] is not None and existing[2] > 0:
                return existing[2]
            if existing[1] == 'not_submitted':
                raise ConfirmedNotSubmitted('Durable intent confirms no submission occurred')
            self.database.execute('UPDATE order_intents SET state=?,result=NULL WHERE remark=?',
                                  ('executing', remark))
            self.database.commit()
            # The broker may not have refreshed its cache yet. Absence is never
            # evidence that an interrupted submission was not accepted.
            result = self._confirmed_order_id(payload)
        else:
            try:
                self._guard_trade(payload, pid)
            except Exception as exc:
                self.database.execute('INSERT INTO order_intents VALUES (?,?,?,NULL)',
                                      (remark, spec, 'not_submitted'))
                self.database.commit()
                raise ConfirmedNotSubmitted(str(exc)) from exc
            # Also check the broker before first submission after DB restoration.
            matches = [o for o in self._orders() if o.order_remark == remark]
            if matches:
                raise UncertainOrder('Broker intent exists without local journal; reconcile first')
            self.database.execute(
                'INSERT INTO order_intents VALUES (?,?,?,NULL)', (remark, spec, 'executing'))
            self.database.commit()
            # The executing marker is durable before the first real API call.
            try:
                self._guard_trade(payload, pid)
            except Exception:
                self.database.execute('UPDATE order_intents SET state=?,result=NULL WHERE remark=?',
                                      ('not_submitted', remark))
                self.database.commit()
                raise ConfirmedNotSubmitted('Final trade guard rejected before broker API')
            result = int(self.client.order_stock(
                self.account, code, payload['order_type'], shares,
                payload['price_type'], payload['price'],
                payload['strategy_name'], remark))
            if result <= 0:
                # This broker can return zero even for a real accepted/filling
                # order. Absence in the cache is not proof of non-submission.
                result = self._confirmed_order_id(payload)
        self.database.execute('UPDATE order_intents SET state=?,result=? WHERE remark=?',
                              ('done', result, remark))
        self.database.commit()
        return result

    def dispatch(self, operation, payload, pid):
        if payload['account_id'] != self.config['account']:
            raise ValueError('Bridge account mismatch')
        if operation == 'asset':
            result = self.client.query_stock_asset(self.account)
        elif operation == 'positions':
            result = self.client.query_stock_positions(self.account)
        elif operation in ('preview_targets', 'execute_targets'):
            from trading.target_file import reconcile_file
            return reconcile_file(self, payload['filename'], pid,
                                  execute=operation == 'execute_targets')
        elif operation == 'orders':
            result = self._orders()
        elif operation == 'order_status':
            result = self.client.query_stock_order(self.account, payload['order_id'])
        elif operation == 'trades':
            result = self.client.query_stock_trades(self.account)
        elif operation == 'bank_transfers':
            result = self.client.query_bank_transfer_stream(
                self.account, payload['start_date'], payload['end_date'])
        elif operation == 'execution_readiness':
            self._owner_check(payload['owner_pid'])
            return {'owner_verified': True, 'live_enabled': self.config['live_enabled'],
                    'order_api_callable': callable(getattr(self.client, 'order_stock', None)),
                    'cancel_api_callable': callable(getattr(self.client, 'cancel_order_stock', None))}
        elif operation == 'unsubmitted_intents':
            confirmed = []
            for request in payload['requests']:
                row = self.database.execute('SELECT specification,state FROM order_intents WHERE remark=?',
                                            (request['remark'],)).fetchone()
                if row is None or row[1] != 'not_submitted':
                    continue
                specification = json.loads(row[0])
                if (specification['account_id'] == self.config['account']
                        and specification['stock_code'] == request['code']
                        and specification['volume'] == request['shares']
                        and specification['order_type'] == (23 if request['direction'] == 'BUY' else 24)
                        and specification['price'] == (0.0 if request['price'] is None else request['price'])
                        and specification['price_type'] == self.config['price_types'][request['price'] is None]):
                    confirmed.append(request['remark'])
            return confirmed
        elif operation == 'submit':
            return self._order(payload, pid)
        elif operation == 'cancel':
            self._guard_trade(payload, pid)
            matches = [o for o in self._orders() if o.order_id == payload['order_id']]
            if len(matches) != 1 or not self._owned_order(matches[0]):
                raise ValueError('Cannot cancel an unowned or unconfirmed order')
            if matches[0].order_status in (53, 54, 56, 57):
                return 0
            self._guard_trade(payload, pid)
            return int(self.client.cancel_order_stock(self.account, payload['order_id']))
        else:
            raise ValueError('Unsupported bridge operation')
        if result is None and operation != 'order_status':
            raise RuntimeError('Broker query returned no response')
        return encode(result)

    def run_one(self):
        row = self.database.execute(
            "SELECT id,expires,client_pid,instance,operation,payload FROM requests "
            "WHERE state='pending' ORDER BY created,id LIMIT 1").fetchone()
        if row is None:
            return False
        request_id, expires, pid, instance, operation, encoded = row
        response = None
        if expires < self.clock():
            response = {'ok': False, 'error': 'Expired request was not executed'}
        elif instance != self.instance:
            response = {'ok': False, 'error': 'Bridge restarted; old request was not executed'}
        if response is None:
            self.database.execute("UPDATE requests SET state='executing' WHERE id=?", (request_id,))
            self.database.commit()
            try:
                self.request_deadline = expires
                result = self.dispatch(operation, json.loads(encoded), pid)
                response = {'ok': True, 'result': result}
            except Exception as exc:
                response = {'ok': False, 'error': type(exc).__name__ + ': ' + str(exc)}
                if isinstance(exc, ConfirmedNotSubmitted):
                    response['execution'] = 'not_submitted'
            finally:
                self.request_deadline = None
        elif operation == 'submit':
            payload = json.loads(encoded)
            remark = payload['order_remark']
            existing = self.database.execute(
                'SELECT state FROM order_intents WHERE remark=?', (remark,)).fetchone()
            if existing is None:
                self.database.execute('INSERT INTO order_intents VALUES (?,?,?,NULL)',
                                      (remark, json.dumps(payload, sort_keys=True, allow_nan=False),
                                       'not_submitted'))
            if existing is None or existing[0] == 'not_submitted':
                response['execution'] = 'not_submitted'
        self.database.execute('UPDATE requests SET state=?,response=? WHERE id=?',
                              ('done', json.dumps(response, allow_nan=False), request_id))
        self.database.commit()
        return True


def serve(directory=None):
    from contextlib import ExitStack
    with ExitStack() as cleanup:
        return _serve(directory, cleanup)


def _serve(directory, cleanup):
    import msvcrt
    import psutil
    from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
    from xtquant.xttype import StockAccount
    root = Path(__file__).resolve().parents[1]
    directory = Path(directory or root / 'data/live_bridge')
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    parent_image = process_image(os.getppid())
    if os.path.normcase(os.path.realpath(parent_image)) not in [
            os.path.normcase(os.path.realpath(path)) for path in config['client_images']]:
        raise RuntimeError('Execution bridge must be launched by the configured QMT client')
    lock = cleanup.enter_context((directory / 'server.lock').open('a+b'))
    lock.seek(0)
    lock.write(b'1')
    lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    database = open_database(directory / 'bridge.sqlite3')
    cleanup.callback(database.close)
    database.execute("UPDATE requests SET state='done',response=? WHERE state='executing'",
                     (json.dumps({'ok': False, 'error': 'Interrupted request; reconcile before retry'}),))
    database.commit()
    events = queue.Queue()

    def flush_events():
        while not events.empty():
            kind, obj, timestamp = events.get_nowait()
            try:
                database.execute('INSERT INTO events(created,kind,payload) VALUES (?,?,?)',
                                 (timestamp, kind, json.dumps(encode(obj), allow_nan=False)))
                database.commit()
            except Exception as exc:
                print('WBR_BRIDGE_AUDIT_WARNING', str(exc), flush=True)

    # ExitStack stops/joins SDK callback workers before this final queue flush,
    # then closes SQLite and releases the bridge lock.
    cleanup.callback(flush_events)

    class Callback(XtQuantTraderCallback):
        def on_stock_order(self, obj):
            events.put(('on_stock_order', obj, time.time()))

        def on_stock_trade(self, obj):
            events.put(('on_stock_trade', obj, time.time()))

        def on_order_error(self, obj):
            events.put(('on_order_error', obj, time.time()))

        def on_cancel_error(self, obj):
            events.put(('on_cancel_error', obj, time.time()))

    client = XtQuantTrader(config['userdata'], int(time.time() * 1000) % 2000000000)
    cleanup.callback(client.stop)
    client.set_timeout(5)
    if config['live_enabled']:
        client.register_callback(Callback())
    client.start()
    if client.connect() != 0:
        raise RuntimeError('QMT bridge connection failed')
    account = StockAccount(config['account'], 'STOCK')
    if client.subscribe(account) != 0:
        raise RuntimeError('QMT bridge account subscription failed')
    bridge = BrokerBridge(config, client, account, database)
    started = psutil.Process(os.getpid()).create_time()
    session_started = time.time()
    heartbeat_at = 0
    print('WBR_BRIDGE_READY', bridge.instance, sys.version.split()[0], os.getpid(), flush=True)
    try:
        while not (directory / 'stop.request').exists():
            reconnect = directory / 'reconnect.request'
            if reconnect.exists():
                reconnect.unlink()
                raise RuntimeError('Requested broker reconnect')
            if time.time() - heartbeat_at >= 1:
                asset = client.query_stock_asset(account)
                if asset is None:
                    raise RuntimeError('Bridge lost its broker account connection')
                heartbeat_at = time.time()
                atomic_json(directory / 'heartbeat.json', {
                    'instance': bridge.instance, 'pid': os.getpid(),
                    'parent_pid': os.getppid(), 'parent_image': parent_image,
                    'executable': sys.executable, 'python_version': sys.version,
                    'started': started, 'updated': heartbeat_at,
                    'session_started': session_started,
                    'source_sha256': globals().get('__source_sha256__', ''),
                    'account': config['account'], 'live_enabled': config['live_enabled'],
                    'connected': True,
                })
            flush_events()
            if not bridge.run_one():
                time.sleep(0.02)
    finally:
        try:
            atomic_json(directory / 'heartbeat.json', {
                'instance': bridge.instance, 'pid': os.getpid(), 'connected': False,
                'updated': time.time(), 'account': config['account']})
        finally:
            print('WBR_BRIDGE_STOPPED', bridge.instance, flush=True)


if __name__ == '__main__':
    serve()
