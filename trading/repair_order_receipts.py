"""Explicit offline reconciliation of recorded intents against fresh broker facts.

No SDK, order, cancellation or target calculation. Production application is
allowed with halted/read-only facts, or explicit audit of a fully completed
plan with exact holdings and only terminal broker orders.
"""
import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import time

from trading.bridge_wire import atomic_json
from trading.target_protocol import broker_fact_date


def reconcile(state, snapshot, specifications, strategy_name, price_types=(11, 44)):
    day = datetime.strptime(state['pending']['trade_date'], '%Y-%m-%d').date()
    if snapshot['account'] != state['account'] or snapshot['trade_date'] != day.isoformat():
        raise ValueError('Account/date mismatch')
    result = copy.deepcopy(state)
    updates, ids = [], set()
    for request in result['requests']:
        if request['direction'] not in ('BUY', 'SELL'):
            raise ValueError('Unknown request direction')
        remark = request['remark']
        matches = [o for o in snapshot['orders'] if o['order_remark'] == remark]
        if len(matches) != 1 or remark not in specifications:
            raise ValueError('No unique broker/durable intent match: ' + remark)
        order, spec = matches[0], specifications[remark]
        side = 23 if request['direction'] == 'BUY' else 24
        if (order['account_id'] != state['account'] or broker_fact_date(order['order_time']) != day
                or order['stock_code'] != request['code'] or order['order_type'] != side
                or order['order_volume'] != request['shares'] or order['strategy_name'] != strategy_name
                or int(order['order_id']) <= 0 or order['order_id'] in ids
                or spec['account_id'] != state['account'] or spec['trade_date'] != day.isoformat()
                or spec['stock_code'] != request['code'] or spec['order_type'] != side
                or spec['volume'] != request['shares'] or spec['strategy_name'] != strategy_name
                or spec['order_remark'] != remark
                or spec['price'] != (0.0 if request['price'] is None else request['price'])
                or spec['price_type'] != price_types[request['price'] is None]
                or (request['price'] is not None and abs(order['price'] - request['price']) > .000001)):
            raise ValueError('Broker/intention mismatch: ' + remark)
        ids.add(order['order_id'])
        request.setdefault('reconciliation_history', []).append(dict(
            previous_order_id=request['order_id'],
            previous_submission_state=request.get('submission_state'),
            previous_rejection_reason=request.get('rejection_reason'),
            broker_order_id=order['order_id'], broker_snapshot_at=snapshot['captured_at']))
        request['order_id'] = int(order['order_id'])
        request.pop('submission_state', None)
        request.pop('rejection_reason', None)
        updates.append((int(order['order_id']), remark))
    return result, updates


def repair(root, apply=False, completed_audit=False):
    root = Path(root).resolve()
    directory = root / 'data/live_file'
    config = json.loads((directory / 'config.json').read_text())
    status = json.loads((directory / 'status.json').read_text())
    snapshot_path = directory / 'account_latest.json'
    snapshot = json.loads(snapshot_path.read_text())
    if (not 0 <= time.time() - snapshot['captured_at'] <= 30
            or not 0 <= time.time() - status['updated'] <= 15
            or snapshot['account'] != config['account']
            or snapshot['trade_date'] != datetime.now().date().isoformat()):
        raise RuntimeError('Fresh read-only facts and trading halt are required')
    path = directory / 'journal' / ('execution_' + snapshot['trade_date'] + '.json')
    original = path.read_bytes()
    state = json.loads(original)
    if completed_audit:
        positions = {p['stock_code']: p['volume'] for p in snapshot['positions']}
        remarks = {r['remark'] for r in state['requests']}
        if (not state['completed'] or status['stage'] != 'completed'
                or any(positions.get(c, 0) != row['shares'] for c, row in state['target_document']['targets'].items())
                or any(o['order_status'] not in (53, 54, 56, 57) for o in snapshot['orders']
                       if o['order_remark'] in remarks)):
            raise RuntimeError('Completed audit requires exact targets and terminal orders')
    elif (config['live_enabled'] or status.get('live_enabled') is not False
            or not (root / 'data/live_trades/trading_watchdog.halt').exists()):
        raise RuntimeError('Active-plan repair requires read-only mode and trading halt')
    db = sqlite3.connect(directory / 'execution.sqlite3')
    try:
        records = db.execute('SELECT remark,specification,state,result FROM order_intents').fetchall()
        specs = {r[0]: json.loads(r[1]) for r in records}
        corrected, updates = reconcile(state, snapshot, specs, config['strategy_name'], config['price_types'])
        target = directory / ('targets_' + snapshot['trade_date'] + '.json')
        if (json.loads(target.read_text()) != state['target_document']
                or hashlib.sha256(target.read_bytes()).hexdigest() != state['target_sha256']):
            raise ValueError('Source target differs from frozen target')
        report = dict(applied=apply, completed_audit=completed_audit, matched=len(updates),
                      journal_sha256_before=hashlib.sha256(original).hexdigest(),
                      snapshot_at=snapshot['captured_at'], updates=updates)
        if apply:
            backup = root / 'tmp/opening_20261103' / ('receipt_repair_' + datetime.now().strftime('%H%M%S_%f'))
            backup.mkdir(parents=True)
            with sqlite3.connect(backup / 'execution.sqlite3') as destination:
                db.backup(destination)
            shutil.copy2(path, backup / path.name)
            shutil.copy2(target, backup / target.name)
            atomic_json(backup / 'account_snapshot.json', snapshot)
            atomic_json(backup / 'old_intents.json', records)
            if path.read_bytes() != original:
                raise RuntimeError('Journal changed during reconciliation')
            with db:
                db.executemany('UPDATE order_intents SET state=\'done\',result=? WHERE remark=?', updates)
            atomic_json(path, corrected)
            report['journal_sha256_after'] = hashlib.sha256(path.read_bytes()).hexdigest()
            atomic_json(backup / 'repair_report.json', report)
            report['backup'] = str(backup)
        return report
    finally:
        db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--completed-audit', action='store_true')
    args = parser.parse_args()
    print(json.dumps(repair(args.root, args.apply, args.completed_audit), ensure_ascii=False))
