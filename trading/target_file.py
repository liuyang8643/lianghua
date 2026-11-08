"""Python 3.6 client-side target-file reconciliation; no strategy imports.

One immutable batch submits each calculated difference at most once. This is
separate from the scheduled OrderMonitor: it never cancels or reprices orders.
"""
import datetime
import hashlib
import json
import math
from pathlib import Path
import re

from trading.bridge_wire import encode
from trading.account_snapshot import validate_account_snapshot
from utils.stock.info import floor_buy_shares, floor_sell_shares


def read_targets(bridge, filename):
    directory = Path(bridge.config['root'], 'data/live_bridge/targets').resolve()
    path = Path(filename).resolve()
    if path.parent != directory or path.suffix != '.json':
        raise ValueError('Target file must be directly inside data/live_bridge/targets')
    raw = path.read_bytes()
    target = json.loads(raw.decode('utf-8-sig'))
    today = datetime.datetime.fromtimestamp(bridge.clock()).date().isoformat()
    if (target['account'] != bridge.config['account'] or target['trade_date'] != today
            or not re.fullmatch(r'[a-f0-9]{12}', target['batch_id'])
            or not math.isfinite(target['expires']) or target['expires'] <= bridge.clock()
            or not isinstance(target['targets'], dict) or not target['targets']):
        raise ValueError('Target account/date/expiry/batch is invalid')
    for code, item in target['targets'].items():
        if (not re.fullmatch(r'\d{6}\.(SH|SZ|BJ)', code)
                or type(item['shares']) is not int or item['shares'] < 0
                or type(item['price']) not in (int, float)
                or not math.isfinite(item['price']) or item['price'] <= 0):
            raise ValueError('Invalid target holding or limit price')
    return target, hashlib.sha256(raw).hexdigest()


def query_positions(bridge):
    asset = bridge.client.query_stock_asset(bridge.account)
    positions = bridge.client.query_stock_positions(bridge.account)
    _, positions = validate_account_snapshot(asset, positions)
    result = {}
    for position in positions:
        code = position.stock_code
        volume, available = position.volume, position.can_use_volume
        if (code in result or type(volume) is not int or type(available) is not int
                or volume < 0 or available < 0 or available > volume):
            raise RuntimeError('Invalid or duplicated broker position')
        result[code] = {'volume': volume, 'available': available}
    return result


def diff_positions(targets, positions, orders):
    """Only explicit symbols are managed; an omitted holding is never sold.

    Pending orders block that symbol, including manual orders and cancel-pending
    orders. Available shares already exclude sell freezes: do not subtract twice.
    """
    result = []
    for code, target in targets.items():
        position = positions.get(code, {'volume': 0, 'available': 0})
        delta = target['shares'] - position['volume']
        pending = 0
        has_pending = False
        has_fill = False
        for order in orders:
            if order.stock_code != code:
                continue
            if order.order_status not in (48, 49, 50, 51, 52, 53, 54, 55, 56, 57):
                raise RuntimeError('Unknown broker order status')
            if order.order_status not in (53, 54, 56, 57):
                has_pending = True
                pending += max(0, order.order_volume - order.traded_volume)
            has_fill = has_fill or order.traded_volume > 0
        shares = 0
        reason = 'at_target'
        if has_pending:
            reason = 'existing_order_wait'
        elif delta and has_fill:
            # Even terminal fills can precede updates to both asset/position
            # caches. A fresh batch must not assume those fills are absorbed.
            reason = 'existing_fill_reconciliation_required'
        elif delta > 0:
            shares = floor_buy_shares(code, delta)
            reason = 'buy' if shares else 'below_minimum'
        elif delta < 0:
            shares = floor_sell_shares(code, -delta, position['available'], position['volume'])
            reason = 'sell' if shares else 'not_sellable_or_below_minimum'
        result.append({'code': code, 'current': position['volume'],
                       'available': position['available'], 'target': target['shares'],
                       'delta': delta, 'pending_shares': pending, 'shares': shares,
                       'direction': ('BUY' if delta > 0 else 'SELL' if delta < 0 else None),
                       'price': target['price'], 'reason': reason})
    return result


def buy(bridge, payload, pid):
    if payload['order_type'] != 23:
        raise ValueError('Buy requires order_type 23')
    return bridge._order(payload, pid)


def sell(bridge, payload, pid):
    if payload['order_type'] != 24:
        raise ValueError('Sell requires order_type 24')
    return bridge._order(payload, pid)


def reconcile_file(bridge, filename, pid, execute=False):
    target, digest = read_targets(bridge, filename)
    original_deadline = bridge.request_deadline
    bridge.request_deadline = (target['expires'] if original_deadline is None
                               else min(original_deadline, target['expires']))
    try:
        return _reconcile_loaded(bridge, target, digest, pid, execute)
    finally:
        bridge.request_deadline = original_deadline


def _reconcile_loaded(bridge, target, digest, pid, execute):
    before = query_positions(bridge)
    orders = bridge._orders()
    if query_positions(bridge) != before:
        raise RuntimeError('Holdings changed while taking snapshot; retry preview')
    differences = diff_positions(target['targets'], before, orders)
    result = {'batch_id': target['batch_id'], 'sha256': digest,
              'dry_run': not execute, 'positions': before,
              'orders': encode(orders), 'diff': differences, 'submissions': []}
    if not execute:
        return result
    if not bridge.config.get('target_file_execution_enabled', False):
        raise RuntimeError('Target-file execution has not been enabled')
    # Do not run a second execution engine alongside the normal daily monitor.
    journal = Path(bridge.config['root'], 'data/live_trades',
                   'execution_' + target['trade_date'] + '.json')
    if journal.exists():
        raise RuntimeError('Daily execution journal exists; target-file mode cannot run concurrently')
    bridge._guard_trade({'trade_date': target['trade_date']}, pid)
    database = bridge.database
    database.execute('CREATE TABLE IF NOT EXISTS target_batches '
                     '(batch_id TEXT PRIMARY KEY, digest TEXT NOT NULL, receipt TEXT NOT NULL)')
    prior = database.execute('SELECT digest,receipt FROM target_batches WHERE batch_id=?',
                             (target['batch_id'],)).fetchone()
    if prior:
        if prior[0] != digest:
            raise RuntimeError('Immutable batch was modified')
        # Do not recompute quantities against possibly lagging post-fill positions.
        result = json.loads(prior[1])
    else:
        if database.execute('SELECT 1 FROM target_batches LIMIT 1').fetchone():
            raise RuntimeError('Prior target batch requires reconciliation before a new batch')
        if any(row['reason'] in ('existing_order_wait', 'existing_fill_reconciliation_required')
               for row in differences):
            raise RuntimeError('Existing orders/fills require reconciliation before freezing batch')
        actions = [row for row in differences if row['shares']]
        actions.sort(key=lambda row: row['direction'] == 'BUY')
        if not actions:
            result['intents'] = []
            result['stage'] = ('at_target' if all(row['delta'] == 0 for row in differences)
                               else 'waiting_for_sellable_or_legal_quantity')
            return result
        result['intents'] = [dict(
            account_id=target['account'], trade_date=target['trade_date'],
            stock_code=row['code'], order_type=23 if row['direction'] == 'BUY' else 24,
            volume=row['shares'], price_type=bridge.config['price_types'][0],
            price=row['price'], strategy_name=bridge.config['strategy_name'],
            order_remark='wbr:{}:{:06d}'.format(target['batch_id'], index + 1))
            for index, row in enumerate(actions)]
        # Freeze the complete batch before the first broker call.
        database.execute('INSERT INTO target_batches VALUES (?,?,?)',
                         (target['batch_id'], digest, json.dumps(result, allow_nan=False)))
        database.commit()
    for payload in result['intents']:
        prior_intent = database.execute('SELECT state FROM order_intents WHERE remark=?',
                                       (payload['order_remark'],)).fetchone()
        if prior_intent is None:
            code = payload['stock_code']
            empty = {'volume': 0, 'available': 0}
            current = query_positions(bridge).get(code, empty)
            if current != result['positions'].get(code, empty):
                raise RuntimeError('Holdings changed since batch freeze; reconcile before submission')
            old_orders = [item for item in result['orders'] if item['stock_code'] == code]
            new_orders = [encode(item) for item in bridge._orders() if item.stock_code == code]
            if new_orders != old_orders:
                raise RuntimeError('Orders changed since batch freeze; reconcile before submission')
        operation = buy if payload['order_type'] == 23 else sell
        order_id = operation(bridge, payload, pid)
        if order_id <= 0:
            raise RuntimeError('Broker rejected target order; reconcile batch before continuing')
        result['submissions'].append({'remark': payload['order_remark'], 'order_id': order_id})
    # Submission acknowledgement is deliberately not called target completion.
    result['stage'] = 'submitted_once_wait_for_broker_fills'
    return result
