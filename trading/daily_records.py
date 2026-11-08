"""Two immutable daily summaries. Python 3.6 stdlib; no SDK/strategy/network."""
from collections import Counter
import datetime
import json
import os
from pathlib import Path
import time

from trading.target_protocol import broker_fact_date, write_immutable_target


def daily_path(directory, producer, day):
    return Path(directory) / 'daily' / (producer + '_' + day + '.json')


def existing_record(path, document):
    if not path.exists():
        return False
    old = json.loads(path.read_text(encoding='utf-8'))
    if (old['account'] != document['account'] or old['trade_date'] != document['trade_date']
            or old['target_id'] != document['id'] or old['target_document'] != document):
        raise ValueError('Daily summary belongs to another frozen target')
    return True


def initial_holdings(document, snapshot=None, journal=None):
    if snapshot is not None:
        return {p['stock_code']: p['volume'] for p in snapshot['positions']}, 'publisher_account_snapshot'
    if 'planning_positions' in document:
        return dict(document['planning_positions']), 'frozen_publisher_positions'
    if journal is None:
        raise ValueError('An original account snapshot or frozen execution plan is required')
    if journal.get('initial_account') is not None:
        return {p['stock_code']: p['volume'] for p in journal['initial_account']['positions']}, 'client_initial_account_snapshot'
    # Older accepted journals did not retain the original account snapshot.
    # Reconstruct only managed share counts; never pretend this is a fresh query.
    before = {c: r['shares'] for c, r in document['targets'].items()}
    for code, quantity in journal['pending']['sell_orders']:
        before[code] += quantity
    for code, quantity in journal['pending']['buy_allocations'].items():
        before[code] -= quantity
    if any(value < 0 for value in before.values()):
        raise ValueError('Invalid frozen-plan reconstruction')
    return before, 'reconstructed_from_frozen_target_and_pending'


def _base(producer, document, simulated):
    return dict(version=1, producer=producer, producer_pid=os.getpid(), recorded_at=time.time(),
                simulated=bool(simulated), account=document['account'], trade_date=document['trade_date'],
                target_id=document['id'], target_document=document)


def write_planner_record(directory, document, snapshot=None, journal=None, simulated=False):
    path = daily_path(directory, 'planner', document['trade_date'])
    if existing_record(path, document):
        return path
    if snapshot is not None and (snapshot['account'] != document['account']
                                 or snapshot['trade_date'] != document['trade_date']):
        raise ValueError('Planner account snapshot does not match target')
    before, source = initial_holdings(document, snapshot, journal)
    report = _base('local_python', document, simulated or document.get('simulation', False))
    report.update(initial_holdings=before, initial_holdings_source=source,
                  initial_account=snapshot, backfilled=snapshot is None)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_immutable_target(path, report)
    return path


def ensure_planner_record(directory, day):
    directory = Path(directory)
    target = directory / ('targets_' + day + '.json')
    journal_path = directory / 'journal' / ('execution_' + day + '.json')
    journal = json.loads(journal_path.read_text(encoding='utf-8')) if journal_path.exists() else None
    if not target.exists() and journal is None:
        return None
    document = json.loads(target.read_text(encoding='utf-8')) if target.exists() else journal['target_document']
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    if document['account'] != config['account'] or document['trade_date'] != day:
        raise ValueError('Daily target account/date mismatch')
    if journal is not None and (journal['account'] != config['account'] or journal['target_document'] != document):
        raise ValueError('Daily journal differs from frozen target')
    if journal is None and 'planning_positions' not in document:
        return None  # Legacy target: wait for evidence; do not invent initial holdings.
    return write_planner_record(directory, document, journal=journal)


def build_executor_record(journal, snapshot, simulated=False):
    document = journal['target_document']
    day = datetime.datetime.strptime(document['trade_date'], '%Y-%m-%d').date()
    if (snapshot['account'] != journal['account'] or document['account'] != journal['account']
            or snapshot['trade_date'] != day.isoformat()
            or datetime.datetime.fromtimestamp(snapshot['captured_at']).date() != day):
        raise ValueError('Execution account/date mismatch')
    before, source = initial_holdings(document, journal=journal)
    rejected = [r for r in journal['requests'] if r.get('submission_state') == 'not_submitted']
    rejected_remarks = {r['remark'] for r in rejected}
    if any(o['order_remark'] in rejected_remarks for o in snapshot['orders'] + (snapshot['trades'] or [])):
        raise ValueError('Non-submission proof conflicts with broker order')
    remarks = {r['remark'] for r in journal['requests']} - rejected_remarks
    requests = {r['remark']: r for r in journal['requests']}
    orders, ids = [], set()
    for order in snapshot['orders']:
        if order['order_remark'] not in remarks:
            continue
        req = requests[order['order_remark']]
        if (broker_fact_date(order['order_time']) != day or order['order_id'] in ids or int(order['order_id']) <= 0
                or order.get('account_id', document['account']) != document['account']
                or order['stock_code'] != req['code'] or order['order_volume'] != req['shares']
                or order['order_type'] != (23 if req['direction'] == 'BUY' else 24)):
            raise ValueError('Broker order conflicts with frozen request')
        ids.add(order['order_id'])
        orders.append(order)
    by_id = {o['order_id']: o for o in orders}
    trades = None
    if snapshot['trades_available'] and snapshot['trades'] is not None:
        unique = {}
        for trade in snapshot['trades']:
            if trade['order_remark'] not in remarks or broker_fact_date(trade['traded_time']) != day:
                continue
            key = (trade['stock_code'], trade['order_type'], trade['traded_id'])
            if not trade['traded_id']:
                raise ValueError('Trade has no unique ID')
            req = requests[trade['order_remark']]
            order = by_id.get(trade['order_id'])
            if (order is None or order['order_remark'] != trade['order_remark']
                    or trade['stock_code'] != req['code']
                    or trade['order_type'] != (23 if req['direction'] == 'BUY' else 24)
                    or trade.get('account_id', document['account']) != document['account']):
                raise ValueError('Trade does not belong to the recorded broker order')
            if key in unique and any(unique[key][f] != trade[f] for f in
                                    ('traded_volume', 'traded_price', 'traded_amount', 'order_id')):
                raise ValueError('Conflicting duplicate trade')
            unique[key] = trade
        trades = list(unique.values())
    actual = {p['stock_code']: p['volume'] for p in snapshot['positions']}
    expected, filled, order_fills = Counter(), Counter(), Counter()
    for code, quantity in journal['pending']['sell_orders']:
        expected[(code, 24)] += quantity
    for code, quantity in journal['pending']['buy_allocations'].items():
        expected[(code, 23)] += quantity
    for order in orders:
        order_fills[(order['stock_code'], order['order_type'])] += order['traded_volume']
    for trade in trades or []:
        filled[(trade['stock_code'], trade['order_type'])] += trade['traded_volume']
    covered = all(actual.get(c, 0) == row['shares'] for c, row in document['targets'].items())
    expected, filled, order_fills = +expected, +filled, +order_fills
    fills_by_id = Counter()
    for trade in trades or []:
        fills_by_id[trade['order_id']] += trade['traded_volume']
    evidence_ok = (trades is not None and filled == order_fills
                   and all(fills_by_id[o['order_id']] == o['traded_volume'] for o in orders))
    complete = bool(covered and evidence_ok and filled == expected
                    and len(orders) == len(remarks)
                    and all(o['order_status'] in (53, 54, 56, 57) for o in orders))
    if journal['completed'] and not complete:
        raise ValueError('Completed execution is not covered by broker holdings/orders/fills')
    report = _base('broker_python', document, simulated)
    report.update(execution_id=journal['id'], state='completed' if complete else 'incomplete',
        journal_completed=journal['completed'], pre_submission_rejections=rejected,
        evidence_verified=evidence_ok, initial_holdings=before, initial_holdings_source=source,
        account_snapshot=snapshot, orders=orders, trades=trades,
        planned_net={c: expected[(c, 23)] - expected[(c, 24)] for c in document['targets']},
        filled_net={c: filled[(c, 23)] - filled[(c, 24)] if evidence_ok else None for c in document['targets']},
        actual_holdings=actual,
        remaining={c: row['shares'] - actual.get(c, 0) for c, row in document['targets'].items()})
    return report


def write_executor_record(directory, journal, snapshot, simulated=False):
    document = journal['target_document']
    path = daily_path(directory, 'executor', document['trade_date'])
    if existing_record(path, document):
        return path
    report = build_executor_record(journal, snapshot, simulated)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_immutable_target(path, report)
    return path
