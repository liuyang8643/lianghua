# coding: utf-8
"""Client-launched readiness evidence. No order/cancel/callback registration."""
import datetime
import json
import os
import sys
import time

ROOT = 'D:/coding/WBR-live/tmp/qmt_readiness_20261101/'


def main():
    with open(ROOT + 'request.json', 'r', encoding='utf-8') as stream:
        request = json.load(stream)
    report = {
        'mode': 'READ_ONLY', 'challenge': request['challenge'],
        'started_at': datetime.datetime.now().isoformat(),
        'executable': sys.executable, 'python_version': sys.version,
        'pid': os.getpid(), 'parent_pid': os.getppid(),
        'modules': {}, 'connection_verified': False,
        'account_snapshot_verified': False,
        'order_or_cancel_calls_in_probe': False,
    }
    client = None
    try:
        for name in ('xtquant', 'xtquant.xttrader', 'xtquant.xttype',
                     'xtquant.xtdata', 'numpy', 'pandas', 'pyarrow'):
            try:
                module = __import__(name, fromlist=['*'])
                report['modules'][name] = {
                    'ok': True, 'file': str(getattr(module, '__file__', '')),
                    'version': str(getattr(module, '__version__', '')),
                }
            except Exception as exc:
                report['modules'][name] = {'ok': False,
                    'error': type(exc).__name__ + ': ' + str(exc)}
        if not report['modules']['xtquant.xttrader']['ok']:
            raise RuntimeError('Broker interpreter cannot import xtquant.xttrader')
        from xtquant.xttrader import XtQuantTrader
        from xtquant.xttype import StockAccount
        client = XtQuantTrader(request['userdata'], int(time.time() * 1000) % 2000000000)
        client.start()
        connected = client.connect()
        report['connect_result'] = connected
        report['connection_verified'] = connected == 0
        if connected != 0:
            raise RuntimeError('Read-only connection failed: ' + str(connected))
        account = StockAccount(request['account'], 'STOCK')
        asset = client.query_stock_asset(account)
        positions = client.query_stock_positions(account)
        orders = client.query_stock_orders(account, cancelable_only=False)
        trades = client.query_stock_trades(account)
        if asset is None or positions is None or orders is None or trades is None:
            raise RuntimeError('Incomplete broker account snapshot')
        report['asset'] = {key: float(getattr(asset, key))
                           for key in ('cash', 'total_asset', 'market_value')}
        report['positions'] = [
            {'code': item.stock_code, 'shares': int(item.volume),
             'sellable': int(item.can_use_volume),
             'market_value': float(item.market_value)}
            for item in positions if item.volume > 0]
        report['position_count'] = len(report['positions'])
        report['order_count'] = len(orders)
        report['trade_count'] = len(trades)
        market_value = sum(item['market_value'] for item in report['positions'])
        report['asset_position_market_value_difference'] = (
            market_value - report['asset']['market_value'])
        report['account_snapshot_verified'] = (
            report['asset']['total_asset'] > 0 and
            abs(report['asset_position_market_value_difference']) <
            max(1.0, report['asset']['market_value'] * 0.01))
    except Exception as exc:
        report['error'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        if client is not None:
            try:
                client.stop()
            except Exception as exc:
                report['stop_error'] = str(exc)
        report['finished_at'] = datetime.datetime.now().isoformat()
        with open(ROOT + 'client_environment.json', 'w', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=True, indent=2, sort_keys=True)
        print('WBR_READONLY_FINISHED', request['challenge'],
              report['connection_verified'], report['account_snapshot_verified'])
        print(json.dumps(report, ensure_ascii=True, sort_keys=True))


if __name__ == '__main__':
    main()
