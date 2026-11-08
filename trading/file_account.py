"""Read-only account facts exported by the broker app. No SDK dependency."""
from datetime import datetime
import json
from pathlib import Path
import time
from types import SimpleNamespace
from collections import Counter

from trading.account_snapshot import validate_account_snapshot
from trading.target_protocol import broker_fact_date


class FileAccount:
    def __init__(self, account, directory='data/live_file', *, trade_date=None, eod=False):
        self.account = account
        self.directory = Path(directory)
        self.trade_date = trade_date
        self.eod = eod
        self._fixed_snapshot = None

    def _snapshot(self):
        if self._fixed_snapshot is not None:
            return self._fixed_snapshot
        name = 'account_' + self.trade_date.isoformat() + '.json' if self.trade_date else 'account_latest.json'
        if self.eod:
            name = 'account_eod_' + self.trade_date.isoformat() + '.json'
        snapshot = json.loads((self.directory / name).read_text(encoding='utf-8'))
        if snapshot['version'] != 1 or snapshot['account'] != self.account:
            raise ValueError('Account facts belong to a different account')
        if self.trade_date:
            if (snapshot['trade_date'] != self.trade_date.isoformat()
                    or datetime.fromtimestamp(snapshot['captured_at']).date() != self.trade_date):
                raise ValueError('Account facts date mismatch')
        elif not 0 <= time.time() - snapshot['captured_at'] <= 30:
            raise ValueError('Client account export is stale')
        validate_account_snapshot(SimpleNamespace(**snapshot['asset']),
                                  [SimpleNamespace(**p) for p in snapshot['positions']])
        if self.trade_date is not None:
            self._fixed_snapshot = snapshot
        return snapshot

    def query_asset(self):
        return SimpleNamespace(**self._snapshot()['asset'])

    def query_positions(self):
        return [SimpleNamespace(**row) for row in self._snapshot()['positions']]

    def query_all_trades(self):
        snapshot = self._snapshot()
        if not snapshot['trades_available'] or snapshot['trades'] is None:
            raise RuntimeError('Client trade facts are unavailable')
        target = self.trade_date or datetime.now().date()
        trades = []
        seen = {}
        for row in snapshot['trades']:
            if broker_fact_date(row['traded_time']) != target:
                continue
            identity = (row['stock_code'], row['order_type'], row['traded_id'])
            economics = tuple(row[field] for field in
                              ('traded_volume', 'traded_price', 'traded_amount', 'order_id'))
            if identity in seen:
                if seen[identity] != economics:
                    raise RuntimeError('Conflicting duplicate trade facts; P&L is not verifiable')
                continue
            if not row['traded_id']:
                raise RuntimeError('Trade fact has no unique trade ID')
            seen[identity] = economics
            trades.append(SimpleNamespace(**row))
        orders = Counter()
        for row in snapshot['orders']:
            if row['traded_volume'] and broker_fact_date(row['order_time']) == target:
                orders[(row['stock_code'], row['order_type'])] += row['traded_volume']
        fills = Counter()
        for row in trades:
            fills[(row.stock_code, row.order_type)] += row.traded_volume
        if orders != fills:
            raise RuntimeError('Broker order fills and trade details disagree; P&L is not verifiable')
        return trades

    def query_bank_transfers(self, start_date, end_date):
        target = self.trade_date or datetime.now().date()
        data = json.loads((self.directory / ('cash_flows_' + target.isoformat() + '.json')).read_text(encoding='utf-8'))
        if data['account'] != self.account:
            raise ValueError('Cash-flow facts belong to a different account')
        return [SimpleNamespace(**row) for row in data['rows']]

    def order(self, *args, **kwargs):
        raise RuntimeError('Local file account cannot submit orders')

    def cancel_order(self, *args, **kwargs):
        raise RuntimeError('Local file account cannot cancel orders')


def get_position(trader, only_can_sell):
    from utils.stock.info import is_convertible_bond
    return [p for p in trader.query_positions()
            if (not only_can_sell or p.can_use_volume > 0) and not is_convertible_bond(p.stock_code)]
