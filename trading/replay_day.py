"""Prepare historical execution inputs from real pre-open receipts and frozen plan.

Local file-only conversion; never recalculates strategy or queries a brokerage.
"""
from datetime import datetime, time as daytime, timedelta
import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace as NS

import pandas as pd

from trading.account_snapshot import validate_account_snapshot
from trading.bridge_wire import atomic_json


def prepare_replay(root, trade_date):
    root = Path(root).resolve()
    config = json.loads((root / 'data/live_file/config.json').read_text(encoding='utf-8'))
    account = config['account']
    plan_path = root / 'data/live_trades' / ('plan_' + trade_date.isoformat() + '.parquet')
    plan = pd.read_parquet(plan_path)
    if plan.empty or not (pd.to_datetime(plan['date']).dt.date == trade_date).all():
        raise ValueError('Frozen plan date does not match replay date')
    before = datetime.combine(trade_date, daytime(9)).timestamp()
    until = datetime.combine(trade_date, daytime(9, 30)).timestamp()
    db = sqlite3.connect('file:' + (root / 'data/live_bridge/bridge.sqlite3').as_posix() + '?mode=ro', uri=True)
    try:
        receipts = db.execute("SELECT id,created,instance,operation,payload,response FROM requests "
            "WHERE created>=? AND created<? AND state='done' AND operation IN ('asset','positions') "
            "ORDER BY created DESC", (before, until)).fetchall()
    finally:
        db.close()
    valid = []
    for request_id, created, instance, operation, payload, response in receipts:
        reply = json.loads(response)
        if json.loads(payload)['account_id'] == account and reply['ok']:
            valid.append(dict(id=request_id, created=created, instance=instance,
                              operation=operation, result=reply['result']))
    pair = None
    for asset in [r for r in valid if r['operation'] == 'asset']:
        candidates = [r for r in valid if r['operation'] == 'positions'
                      and r['instance'] == asset['instance'] and abs(r['created'] - asset['created']) <= 10]
        if candidates:
            positions = min(candidates, key=lambda r: abs(r['created'] - asset['created']))
            pair = asset, positions
            break
    if pair is None:
        raise RuntimeError('No account-bound pre-open broker receipt pair; cannot fabricate historical holdings')
    asset, position_receipt = pair
    holdings = [p for p in position_receipt['result'] if p['volume'] > 0]
    if asset['result']['account_id'] != account or any(p['account_id'] != account for p in holdings):
        raise ValueError('Historical account attribution mismatch')
    validate_account_snapshot(NS(**asset['result']), [NS(**p) for p in holdings])
    by_code = {p['stock_code']: p for p in holdings}
    prices, names, targets, expected = {}, {}, {}, []
    for row in plan.to_dict('records'):
        code, quantity = row['code'], int(row['est_volume'])
        price = float(row['est_price'])
        if not price > 0:
            raise ValueError('Original plan has no executable open price')
        prices[code], names[code] = price, str(row['name'])
        targets.setdefault(code, int(by_code[code]['volume']) if code in by_code else 0)
        if quantity:
            if row['direction'] == 'sell' and quantity < 0:
                quantity = int(by_code[code]['can_use_volume'])
            if quantity <= 0 or row['direction'] not in ('buy', 'sell'):
                raise ValueError('Original plan direction/quantity is invalid')
            sign = 1 if row['direction'] == 'buy' else -1
            targets[code] += sign * quantity
            expected.append(dict(code=code, direction=row['direction'].upper(), shares=quantity, price=price))
    if any(shares < 0 for shares in targets.values()):
        raise ValueError('Original plan would oversell its historical account')
    from core.rebalance import freeze_unit_price
    from utils.stock.time import get_last_trading_day
    previous = get_last_trading_day(trade_date - timedelta(days=1))
    freeze, freeze_sources = {}, {}
    for code, price in prices.items():
        path = root / 'data/k-line' / (code + '.parquet')
        frame = pd.read_parquet(path, columns=['time', 'close'])
        dates = (pd.to_datetime(frame['time'], unit='ms') if pd.api.types.is_numeric_dtype(frame['time'])
                 else pd.to_datetime(frame['time']))
        rows = frame.loc[dates.dt.date == previous]
        if rows.empty:
            raise ValueError('Missing actual T-1 close for freeze reference: ' + code)
        close = float(rows.iloc[-1]['close'])
        freeze[code] = float(freeze_unit_price(code, price, close))
        freeze_sources[code] = dict(date=previous.isoformat(), close=close)
    priority = plan.loc[plan['direction'] == 'buy'].sort_values('plan_seq')['code'].tolist()
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    captured = max(asset['created'], position_receipt['created'])
    published = max(captured, plan_path.stat().st_mtime)
    if not before <= published < until:
        raise ValueError('Original plan creation time is not a verified pre-open time')
    document = dict(version=1, id=digest[:12], account=account, trade_date=trade_date.isoformat(),
        generated_at=published, expires=datetime.combine(trade_date, daytime(15)).timestamp(),
        targets={c: dict(shares=q, open_price=prices[c], freeze_price=freeze[c]) for c, q in targets.items()},
        buy_priority=priority)
    fixture = dict(version=1, simulation_only=True, account=account, trade_date=trade_date.isoformat(),
        source_plan=str(plan_path), source_plan_sha256=digest,
        source_receipts=[asset['id'], position_receipt['id']], source_instance=asset['instance'],
        account_captured_at=captured, asset=asset['result'], positions=holdings,
        document=document, names=names, expected_orders=expected,
        freeze_price_sources=freeze_sources,
        note='Frozen original plan converted to absolute targets; no factor recalculation; no historical fills fabricated')
    directory = root / 'data/live_file/replays' / trade_date.isoformat()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'input.json'
    if path.exists():
        old = json.loads(path.read_text(encoding='utf-8'))
        if old != fixture:
            raise RuntimeError('Historical replay input is immutable; original evidence changed')
    else:
        atomic_json(path, fixture)
    return path


def install_replay_entry(root, trade_date):
    from trading.bridge_setup import bootstrap_source
    root = Path(root).resolve()
    fixture = prepare_replay(root, trade_date)
    config = json.loads((root / 'data/live_file/config.json').read_text(encoding='utf-8'))
    entry = Path(config['broker_python']).parent.parent / 'mpython' / (
        'WBR_REPLAY_' + trade_date.strftime('%Y%m%d') + '.py')
    source = bootstrap_source(root, 'trading/replay_worker.py')
    source = source.replace('import sys\n',
        'import sys\nimport os\nos.environ[\'WBR_REPLAY_INPUT\'] = %r\n' % str(fixture), 1)
    entry.write_text(source, encoding='utf-8')
    return entry
