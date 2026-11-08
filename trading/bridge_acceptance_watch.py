"""Observe the existing strategy's real opening orders; never place test orders."""
import argparse
from datetime import date, datetime, time as wall_time
import json
import os
from pathlib import Path
import time

from trading.bridge_client import BridgeUnavailable
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
    from configs import TRADE_ACCOUNT
    from trading.trader import Trader
    root = Path(__file__).resolve().parents[1]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    execution = root / 'data/live_trades' / ('execution_' + trade_date.isoformat() + '.json')
    identity_path = root / 'data/live_trades/trading_main.identity.json'
    deadline = datetime.combine(trade_date, until)
    trader = None
    previous_stage = None
    status = {'trade_date': trade_date.isoformat(), 'success': False,
              'read_only': True, 'observer_pid': os.getpid(), 'actual_order_or_cancel_calls': 0}
    try:
        while datetime.now() < deadline:
            status = {'checked_at': datetime.now().isoformat(), 'trade_date': trade_date.isoformat(),
                      'read_only': True, 'observer_pid': os.getpid(),
                      'actual_order_or_cancel_calls': 0, 'success': False}
            try:
                if trader is None:
                    trader = Trader(TRADE_ACCOUNT, backend='bridge',
                                    enable_callbacks=False, enable_orders=False)
                heartbeat = trader.client.ensure_alive()
                identity = json.loads(identity_path.read_text(encoding='utf-8'))
                owner = trader.client.verify_live_owner(identity['pid'])
                if not (owner.owner_verified and owner.live_enabled):
                    raise ValueError('Live main/lock/client execution mode is not ready')
                status.update(bridge_pid=heartbeat['pid'], main_pid=identity['pid'],
                              main_owner_verified=True,
                              broker_source_sha256=heartbeat['source_sha256'])
                if not execution.exists():
                    status['stage'] = 'WAITING_FOR_STRATEGY_PLAN'
                else:
                    journal = json.loads(execution.read_text(encoding='utf-8'))
                    if (journal['account'] != TRADE_ACCOUNT
                            or journal['pending']['trade_date'] != trade_date.isoformat()):
                        raise ValueError('Execution journal account/date mismatch')
                    if journal['completed'] and not journal['requests']:
                        status['stage'] = 'NO_ACTIONS_PLANNED_ORDER_PERMISSION_UNPROVEN'
                        atomic_json(output / 'opening_watch_status.json', status)
                        return 2
                    if not journal['requests']:
                        status['stage'] = 'WAITING_FOR_CONTINUOUS_TRADING'
                    else:
                        orders = trader.query_orders()
                        trades = trader.query_all_trades()
                        accepted, fills = matched_broker_facts(journal, orders, trades)
                        remarks = {r['remark'] for r in journal['requests']}
                        rejected = [dict(order_id=o.order_id, code=o.stock_code, reason=o.status_msg)
                                    for o in orders if o.order_remark in remarks and o.order_status == 57]
                        status.update(accepted_orders=accepted, matched_fills=fills,
                                      broker_rejections=rejected, journal_completed=journal['completed'])
                        if accepted and fills:
                            asset, positions = trader.query_account_snapshot()
                            owner = trader.client.verify_live_owner(identity['pid'])
                            if not owner.owner_verified:
                                raise ValueError('Current live main identity not confirmed')
                            coverage = execution_coverage(journal, fills)
                            complete = bool(journal['completed'] and coverage['complete'])
                            status.update(stage=('REAL_PLAN_EXECUTION_VERIFIED' if complete
                                                 else 'REAL_CHAIN_VERIFIED_PLAN_PENDING'),
                                          success=complete, chain_verified=True,
                                          actual_order_permission_verified=True, target_coverage=coverage,
                                          evidence_source='BROKER_QUERIES_VIA_CLIENT_PYTHON',
                                          position_count=len(positions), total_asset=asset.total_asset)
                            atomic_json(output / 'opening_execution_evidence.json', status)
                            atomic_json(output / 'opening_watch_status.json', status)
                            if complete:
                                print(status['stage'], len(accepted), 'orders', len(fills), 'fills', flush=True)
                                return 0
                        else:
                            status['stage'] = ('BROKER_ACCEPTED_WAITING_FOR_FILL' if accepted
                                               else 'WAITING_FOR_BROKER_ACCEPTANCE')
            except (BridgeUnavailable, Exception) as exc:
                status.update(stage='WAITING_FOR_RECOVERY', error=type(exc).__name__ + ': ' + str(exc))
                if trader is not None:
                    trader.client.stop()
                    trader = None
            atomic_json(output / 'opening_watch_status.json', status)
            if previous_stage != status['stage']:
                print(status['checked_at'], status['stage'], flush=True)
                previous_stage = status['stage']
            time.sleep(interval)
        status.update(stage='DEADLINE_WITHOUT_VERIFIED_FILL', success=False,
                      checked_at=datetime.now().isoformat())
        atomic_json(output / 'opening_watch_status.json', status)
        return 1
    finally:
        if trader is not None:
            trader.client.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trade-date', type=date.fromisoformat, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(watch(args.trade_date, args.output))
