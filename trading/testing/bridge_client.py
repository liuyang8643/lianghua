"""Test-only RPC harness for broker receipt regression; no production entry."""
from datetime import date
import hashlib
import json
import os
import operator
from pathlib import Path
import sqlite3
import sys
import time
from types import SimpleNamespace
import uuid

import psutil

from trading.bridge_wire import BridgeUnavailable, ConfirmedNotSubmitted, atomic_json, open_database


def decode(value):
    if isinstance(value, list):
        return [decode(item) for item in value]
    if isinstance(value, dict):
        return SimpleNamespace(**{key: decode(item) for key, item in value.items()})
    return value


class BridgeClient:
    def __init__(self, account_id, directory=None, *, timeout=12.0):
        self.directory = Path(directory or Path(__file__).resolve().parents[2] / 'data/live_bridge')
        self.account_id = account_id
        self.timeout = timeout
        self.callback = None
        self.database = None
        self.instance = None
        self.live_enabled = False
        self.expected_source_hash = hashlib.sha256(
            (Path(__file__).resolve().parents[1] / 'qmt_bridge.py').read_text(encoding='utf-8').encode('utf-8')).hexdigest()
        self.cursor_path = self.directory / 'events_cursor.json'
        self.event_cursor = 0

    def register_callback(self, callback):
        self.callback = callback
        if self.cursor_path.exists():
            try:
                cursor = json.loads(self.cursor_path.read_text(encoding='utf-8'))['seq']
                if type(cursor) is not int or cursor < 0:
                    raise ValueError('Audit cursor must be a nonnegative integer')
                self.event_cursor = cursor
            except (OSError, ValueError, KeyError, TypeError) as exc:
                from trading.logger import trading_logger
                trading_logger.warning(f'Bridge audit cursor unreadable; replaying durable events: {exc}')

    def set_relaxed_response_order_enabled(self, enabled):
        pass  # Local RPC is serialized, so no SDK response-order thread is needed.

    def start(self):
        self.database = open_database(self.directory / 'bridge.sqlite3')

    def _heartbeat(self):
        try:
            state = json.loads((self.directory / 'heartbeat.json').read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise RuntimeError('QMT execution bridge is not running in the client') from exc
        if (not state['connected'] or state['account'] != self.account_id
                or state['source_sha256'] != self.expected_source_hash
                or not 0 <= time.time() - state['updated'] <= 10):
            raise RuntimeError('QMT bridge heartbeat is stale/disconnected/account mismatch')
        process = psutil.Process(state['pid'])
        parent = psutil.Process(state['parent_pid'])
        if (os.path.normcase(process.exe()) != os.path.normcase(state['executable'])
                or process.ppid() != parent.pid
                or os.path.normcase(parent.exe()) != os.path.normcase(state['parent_image'])
                or abs(process.create_time() - state['started']) > 0.1):
            raise RuntimeError('QMT bridge process identity mismatch')
        return state

    def connect(self):
        state = self._heartbeat()
        self.instance = state['instance']
        self.live_enabled = state['live_enabled']
        return 0

    def ensure_alive(self):
        try:
            state = self._heartbeat()
        except (RuntimeError, OSError, ValueError, KeyError, psutil.Error) as exc:
            raise BridgeUnavailable('QMT bridge unavailable; restart and reconcile: ' + str(exc)) from exc
        if (self.instance != state['instance']
                or (self.live_enabled and not state['live_enabled'])):
            raise BridgeUnavailable('QMT bridge changed; rebuild Trader and reconcile execution journal')
        self._drain_events()
        return state

    def subscribe(self, account):
        if account.account_id != self.account_id:
            raise ValueError('Bridge account mismatch')
        return 0

    def stop(self):
        if self.database is not None:
            self.database.close()
            self.database = None

    def _drain_events(self):
        if self.callback is None:
            return
        from trading.logger import trading_logger
        try:
            rows = self.database.execute('SELECT seq,created,kind,payload FROM events WHERE seq>? ORDER BY seq',
                                         (self.event_cursor,)).fetchall()
        except sqlite3.Error as exc:
            trading_logger.warning(f'Bridge audit event read failed; trading continues: {exc}')
            return
        for seq, created, kind, payload in rows:
            try:
                event = decode(json.loads(payload))
                event._bridge_event_time = created
                getattr(self.callback, kind)(event)
            except Exception as exc:
                trading_logger.warning(f'Bridge event audit failed: {exc}')
            self.event_cursor = seq
        if rows:
            try:
                atomic_json(self.cursor_path, {'seq': self.event_cursor})
            except OSError as exc:
                trading_logger.warning(f'Bridge event cursor save failed: {exc}')

    def _rpc(self, operation, **payload):
        try:
            state = self.ensure_alive()
        except BridgeUnavailable as exc:
            # This request has not been inserted into the durable queue yet.
            exc.not_submitted = operation == 'submit'
            raise
        if operation in ('submit', 'cancel') and not state['live_enabled']:
            raise BridgeUnavailable('QMT execution bridge is read-only',
                                    not_submitted=operation == 'submit')
        payload['account_id'] = self.account_id
        request_id = uuid.uuid4().hex
        now = time.time()
        self.database.execute('INSERT INTO requests VALUES (?,?,?,?,?,?,?,?,NULL)',
                              (request_id, now, now + self.timeout, os.getpid(), self.instance,
                               operation, json.dumps(payload, allow_nan=False), 'pending'))
        self.database.commit()
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            row = self.database.execute('SELECT state,response FROM requests WHERE id=?',
                                        (request_id,)).fetchone()
            if row[0] == 'done':
                response = json.loads(row[1])
                self._drain_events()
                if not response['ok']:
                    if response.get('execution') == 'not_submitted':
                        raise ConfirmedNotSubmitted(response['error'])
                    raise RuntimeError(response['error'])
                return decode(response['result'])
            time.sleep(0.025)
        raise RuntimeError(f'Bridge {operation} timed out; state uncertain, do not resend')

    def query_stock_asset(self, account):
        return self._rpc('asset')

    def verify_live_owner(self, owner_pid):
        return self._rpc('execution_readiness', owner_pid=owner_pid)

    def query_not_submitted_intents(self, requests):
        return self._rpc('unsubmitted_intents', requests=requests)

    def query_stock_positions(self, account):
        return self._rpc('positions')

    def query_stock_orders(self, account, cancelable_only=False):
        orders = self._rpc('orders')
        if cancelable_only:
            return [o for o in orders if o.order_status in (48, 49, 50, 51, 52, 55)]
        return orders

    def query_stock_order(self, account, order_id):
        return self._rpc('order_status', order_id=order_id)

    def query_stock_trades(self, account):
        return self._rpc('trades')

    def query_bank_transfer_stream(self, account, start_date, end_date):
        return self._rpc('bank_transfers', start_date=start_date, end_date=end_date)

    def order_stock(self, account, stock_code, order_type, order_volume,
                    price_type, price, strategy_name, order_remark):
        if isinstance(order_volume, bool):
            raise ValueError('Order volume must be a whole number of shares')
        order_volume = operator.index(order_volume)
        return self._rpc('submit', stock_code=stock_code, order_type=int(order_type),
                         volume=order_volume, price_type=int(price_type), price=float(price),
                         strategy_name=strategy_name, order_remark=order_remark,
                         trade_date=date.today().isoformat())

    def cancel_order_stock(self, account, order_id):
        return self._rpc('cancel', order_id=int(order_id), trade_date=date.today().isoformat())
