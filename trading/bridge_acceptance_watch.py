"""Observe the existing strategy's real opening orders; never place test orders."""
import argparse
from datetime import date, datetime, time as wall_time
import json
import os
from pathlib import Path
import time

from trading.bridge_wire import atomic_json


def matched_broker_facts(journal, orders, trades):
    requests = {item['remark']: item for item in journal['requests']}
    matched = {}
    for order in orders:
        request = requests.get(order.order_remark)
        if request is None:
            continue
        if request.get('submission_state') == 'not_submitted':
            raise ValueError('Broker order contradicts a no-submission proof')
        if (order.stock_code != request['code'] or order.order_volume != request['shares']
                or order.order_type != (23 if request['direction'] == 'BUY' else 24)):
            raise ValueError('Broker order differs from the saved execution intent')
        if order.order_id > 0 and order.order_status in (50, 51, 52, 53, 54, 55, 56):
            matched[order.order_id] = order
    accepted = [{key: getattr(order, key, None) for key in (
        'order_id', 'stock_code', 'order_type', 'order_volume', 'traded_volume',
        'order_status', 'order_remark', 'order_sysid', 'price_type', 'price')}
        for order in matched.values()]
    fills = []
    seen = set()
    quantities = {}
    for trade in trades:
        order = matched.get(trade.order_id)
        if order is None:
            continue
        if (trade.stock_code != order.stock_code or trade.order_type != order.order_type
                or trade.traded_volume <= 0 or trade.traded_price <= 0):
            raise ValueError('Broker fill does not match the accepted order')
        key = trade.traded_id or (trade.order_id, trade.traded_time,
                                 trade.traded_volume, trade.traded_price)
        if key in seen:
            continue
        seen.add(key)
        quantities[trade.order_id] = quantities.get(trade.order_id, 0) + trade.traded_volume
        if quantities[trade.order_id] > order.order_volume:
            raise ValueError('Queried fills exceed the accepted order quantity')
        fills.append({key: getattr(trade, key, None) for key in (
            'order_id', 'stock_code', 'order_type', 'traded_id', 'traded_time',
            'traded_volume', 'traded_price', 'traded_amount')})
    return accepted, fills


def execution_coverage(journal, fills):
    targets = {}
    for code, shares in journal['pending']['buy_allocations'].items():
        if shares > 0:
            targets[('BUY', code)] = int(shares)
    for code, shares in journal['pending']['sell_orders']:
        if shares < 0:
            raise ValueError('Execution journal still contains an unresolved liquidation target')
        if shares > 0:
            targets[('SELL', code)] = targets.get(('SELL', code), 0) + int(shares)
    quantities = {}
    for item in fills:
        key = ('BUY' if item['order_type'] == 23 else 'SELL', item['stock_code'])
        quantities[key] = quantities.get(key, 0) + int(item['traded_volume'])
    if any(amount > 0 and key not in targets for key, amount in quantities.items()):
        raise ValueError('Broker fills contain a trade outside the original execution targets')
    rows = []
    for (direction, code), target in sorted(targets.items()):
        filled = quantities.get((direction, code), 0)
        if filled > target:
            raise ValueError('Broker fills exceed the original execution target')
        rows.append(dict(direction=direction, code=code, target=target,
                         filled=filled, remaining=target - filled))
    return {'complete': all(item['remaining'] == 0 for item in rows), 'targets': rows}


def watch(trade_date, output, *, until=wall_time(10, 5), interval=30):
    """Observe exported broker facts only; never connect a local SDK or RPC."""
    from trading.file_account import FileAccount
    root=Path(__file__).resolve().parents[1];directory=root/'data/live_file'
    config=json.loads((directory/'config.json').read_text(encoding='utf8'))
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    execution=directory/'journal'/('execution_'+trade_date.isoformat()+'.json')
    deadline=datetime.combine(trade_date,until);previous_stage=None
    status=dict(trade_date=trade_date.isoformat(),success=False,read_only=True,
                observer_pid=os.getpid(),actual_order_or_cancel_calls=0)
    while datetime.now()<deadline:
        status.update(checked_at=datetime.now().isoformat(),success=False)
        try:
            account=FileAccount(config['account'],directory)
            facts=account._snapshot()
            if facts['trade_date']!=trade_date.isoformat():raise ValueError('Broker facts belong to another day')
            account._fixed_snapshot=facts
            if not execution.exists():status['stage']='WAITING_FOR_STRATEGY_PLAN'
            else:
                journal=json.loads(execution.read_text(encoding='utf8'))
                if (journal['account']!=config['account'] or journal['pending']['trade_date']!=trade_date.isoformat()
                        or journal['target_document'].get('simulation',False)):
                    raise ValueError('Execution identity or simulation flag is invalid')
                if journal['completed'] and not journal['requests']:
                    status['stage']='NO_ACTIONS_PLANNED_ORDER_PERMISSION_UNPROVEN'
                    atomic_json(output/'opening_watch_status.json',status);return 2
                accepted,fills=matched_broker_facts(journal,account.query_orders(),account.query_all_trades())
                coverage=execution_coverage(journal,fills)
                complete=bool(accepted and fills and journal['completed'] and coverage['complete'])
                status.update(accepted_orders=accepted,matched_fills=fills,execution_coverage=coverage,
                    success=complete,stage='REAL_EXECUTION_COMPLETED' if complete else 'WAITING_FOR_BROKER_EXECUTION')
                if complete:
                    atomic_json(output/'opening_watch_status.json',status);return 0
        except Exception as error:
            status.update(stage='WAITING_FOR_VERIFIED_BROKER_FILES',error=type(error).__name__+': '+str(error))
        atomic_json(output/'opening_watch_status.json',status)
        if previous_stage!=status['stage']:
            print(status['stage'],flush=True);previous_stage=status['stage']
        time.sleep(interval)
    status.update(stage='DEADLINE_WITHOUT_VERIFIED_FILL',success=False)
    atomic_json(output/'opening_watch_status.json',status);return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trade-date', type=date.fromisoformat, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(watch(args.trade_date, args.output))
