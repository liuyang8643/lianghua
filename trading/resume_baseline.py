"""Explicit, dated recovery budget from today's real pre-open account facts.

Never manufactures a previous-day EOD record or starts a trading process.
"""
import argparse
from datetime import date, datetime, time as daytime
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from trading.account_snapshot import validate_account_snapshot
from trading.target_protocol import (TERMINAL_ORDER_STATUSES, broker_fact_date,
                                     read_document, write_immutable_target)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def prepare_request(root, trade_date, *, now=None):
    root = Path(root).resolve()
    now = now or datetime.now()
    directory = root / 'data/live_file'
    config, _ = read_document(directory / 'config.json')
    if trade_date != now.date() or now.time() >= daytime(10):
        raise ValueError('Resume request is only valid for today before 10:00')
    if trade_date.isoformat() not in config['trading_dates']:
        raise ValueError('Resume date is not a verified trading date')
    for path in (directory / f'targets_{trade_date}.json',
                 directory / 'journal' / f'execution_{trade_date}.json'):
        if path.exists():
            raise RuntimeError('A frozen plan already exists; resume its journal instead')
    request = dict(version=1, account=config['account'], trade_date=trade_date.isoformat(),
                   source='explicit_same_day_account_resume', root=str(root),
                   strategy_sha256=hashlib.sha256((root / 'configs/config.json').read_bytes()).hexdigest(),
                   expires=datetime.combine(trade_date, daytime(10)).timestamp())
    path = directory / f'resume_request_{trade_date}.json'
    write_immutable_target(path, request)
    return path


def load_resume_budget(directory, snapshot, strategy_path, now):
    directory = Path(directory).resolve()
    root = directory.parents[1]
    today = now.date()
    request_path = directory / f'resume_request_{today}.json'
    request, request_hash = read_document(request_path)
    if (request['version'] != 1 or request['source'] != 'explicit_same_day_account_resume'
            or request['account'] != snapshot['account'] or request['trade_date'] != today.isoformat()
            or Path(request['root']).resolve() != root
            or request['strategy_sha256'] != hashlib.sha256(Path(strategy_path).read_bytes()).hexdigest()
            or not now.timestamp() < request['expires']
            or not daytime(9, 25, 10) <= now.time() < daytime(9, 30)):
        raise ValueError('Recovery budget requires a matching pre-open resume request')
    if (snapshot['version'] != 1 or snapshot['trade_date'] != today.isoformat()
            or not 0 <= now.timestamp() - snapshot['captured_at'] <= 30):
        raise ValueError('Recovery account facts are stale, future or from another date')
    validate_account_snapshot(SimpleNamespace(**snapshot['asset']),
                              [SimpleNamespace(**p) for p in snapshot['positions']])
    if (snapshot['asset']['account_id'] != snapshot['account']
            or any(p['account_id'] != snapshot['account'] for p in snapshot['positions'])
            or snapshot.get('simulation', False)):
        raise ValueError('Recovery snapshot account attribution is invalid')
    if (snapshot['trades_available'] is not True
            or not isinstance(snapshot['trades'], list) or not isinstance(snapshot['orders'], list)):
        raise ValueError('Recovery requires complete order/trade queries')
    # Before the opening auction ends, reject any same-day activity or unknown
    # timestamps; a recovery budget must not silently absorb an execution.
    for records, field in ((snapshot['orders'], 'order_time'), (snapshot['trades'], 'traded_time')):
        if any(broker_fact_date(row[field]) >= today for row in records):
            raise ValueError('Same-day orders/trades require reconciliation before recovery')
        if any(row['account_id'] != snapshot['account'] for row in records):
            raise ValueError('Order/trade facts belong to another account')
    if any(row['order_status'] not in TERMINAL_ORDER_STATUSES for row in snapshot['orders']):
        raise ValueError('Active or unknown orders require reconciliation before recovery')
    frozen_cash = snapshot['asset']['frozen_cash']
    if type(frozen_cash) not in (int, float) or not math.isfinite(frozen_cash) or frozen_cash != 0:
        raise ValueError('Frozen cash must be a verified finite zero before recovery')
    path = directory / f'budget_baseline_{today}.json'
    if path.exists():
        saved, digest = read_document(path)
        if (saved['request_sha256'] != request_hash or saved['source'] != request['source']
                or saved['account'] != snapshot['account'] or saved['trade_date'] != today.isoformat()
                or saved['snapshot_canonical_sha256'] != _digest(saved['snapshot'])):
            raise ValueError('Frozen recovery budget identity or contents changed')
        original = saved['snapshot']
        old_holdings = {p['stock_code']: p['volume'] for p in original['positions']}
        current_holdings = {p['stock_code']: p['volume'] for p in snapshot['positions']}
        if old_holdings != current_holdings or original['asset']['cash'] != snapshot['asset']['cash']:
            raise ValueError('Account changed after freezing recovery budget; reconcile first')
    else:
        saved = dict(version=1, source=request['source'], account=snapshot['account'],
                     trade_date=today.isoformat(), captured_at=snapshot['captured_at'],
                     request_sha256=request_hash, snapshot=snapshot,
                     snapshot_canonical_sha256=_digest(snapshot), missing_period_pnl_verified=False)
        write_immutable_target(path, saved)
        saved, digest = read_document(path)
    original = saved['snapshot']
    metadata = dict(source=saved['source'], trade_date=saved['trade_date'],
                    captured_at=saved['captured_at'], path=path.name, sha256=digest,
                    missing_period_pnl_verified=False)
    return (original['asset']['cash'],
            {p['stock_code']: p['volume'] for p in original['positions']}, metadata)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', required=True, type=date.fromisoformat)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(prepare_request(args.root, args.date))
