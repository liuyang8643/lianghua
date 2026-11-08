"""Python 3.6 execution simulation. NEVER constructs a real brokerage client."""
import datetime
import importlib.abc
import json
import logging
import os
from pathlib import Path
import socket
import sys
import tempfile
import traceback
from types import SimpleNamespace as NS
from trading.target_protocol import FILE_EXECUTOR_PROTOCOL


class DenyLiveBroker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('xtquant.xttrader', 'xtquant.xtdata', 'xtquant.xtdatacenter')):
            raise RuntimeError('Real brokerage module prohibited in simulation: ' + fullname)
        if fullname.split('.')[0] in ('numpy','pandas','lightgbm','factor','core','env','strategy_release'):
            raise RuntimeError('Client execution must not import strategy/data frameworks: ' + fullname)


def deny_network(*args, **kwargs):
    raise RuntimeError('Network access prohibited in app simulation')


class Clock:
    def __init__(self, day, hour=9, minute=29, second=59):
        self.value = datetime.datetime.combine(day, datetime.time(hour, minute, second)).timestamp()

    def time(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds

    def now(self):
        return datetime.datetime.fromtimestamp(self.value)


class FakeSDK:
    def __init__(self, clock, holdings, cash, scenario):
        self.clock = clock
        self.holdings = dict(holdings)
        self.sellable = dict(holdings)
        self.cash = cash
        self.scenario = scenario
        self.orders = []
        self.calls = []
        self.cancels = []
        self.rejected = False
        self.crashed = False
        self.fake_trades = []

    def price(self, code):
        return 20.0 if code == '600002.SH' else 10.0

    def fill(self, order, quantity):
        if quantity <= 0:
            return
        code = order.stock_code
        sign = 1 if order.order_type == 23 else -1
        self.holdings[code] = self.holdings.get(code, 0) + sign * quantity
        if sign < 0:
            self.sellable[code] -= quantity
        self.cash -= sign * quantity * self.price(code)
        order.traded_volume += quantity
        order.traded_price = self.price(code)
        order.order_status = 56 if order.traded_volume == order.order_volume else 55
        self.fake_trades.append(NS(account_id='SIMULATED', stock_code=code, order_type=order.order_type,
            order_id=order.order_id, order_remark=order.order_remark,
            traded_id='FAKE%06d' % (len(self.fake_trades) + 1), traded_volume=quantity,
            traded_price=self.price(code), traded_amount=quantity * self.price(code), traded_time=self.clock.time()))

    def update(self):
        for order in self.orders:
            if order.order_status in (53, 54, 56, 57):
                continue
            elapsed = self.clock.time() - order.order_time
            if self.scenario in ('keep_small_kcb_buy', 'keep_odd_main_buy', 'keep_small_kcb_sell'):
                if elapsed >= 65:
                    self.fill(order, order.order_volume - order.traded_volume)
                elif elapsed >= 2 and order.traded_volume == 0:
                    first = {'keep_small_kcb_buy': 50, 'keep_odd_main_buy': 150,
                             'keep_small_kcb_sell': 300}[self.scenario]
                    self.fill(order, first)
            elif self.scenario == 'partial_reprice' and order.price_type == 11 and order.order_type == 23:
                if elapsed >= 2 and order.traded_volume == 0:
                    self.fill(order, 100)
            elif elapsed >= 2:
                self.fill(order, order.order_volume - order.traded_volume)

    def query_stock_asset(self, account):
        self.update()
        value = sum(shares * self.price(code) for code, shares in self.holdings.items())
        frozen = sum((o.order_volume - o.traded_volume) * self.price(o.stock_code)
                     for o in self.orders if o.order_type == 23 and o.order_status not in (53, 54, 56, 57))
        return NS(account_id='SIMULATED', cash=max(0, self.cash - frozen), market_value=value,
                  total_asset=self.cash + value)

    def query_stock_positions(self, account):
        self.update()
        return [NS(account_id='SIMULATED', stock_code=code, volume=shares,
                   can_use_volume=min(shares, self.sellable.get(code, 0)),
                   market_value=shares * self.price(code)) for code, shares in self.holdings.items() if shares]

    def query_stock_orders(self, account, cancelable_only=False):
        self.update()
        return self.orders

    def query_stock_order(self, account, order_id):
        self.update()
        return next((order for order in self.orders if order.order_id == order_id), None)

    def query_stock_trades(self, account):
        return self.fake_trades

    def order_stock(self, account, code, side, volume, price_type, price, strategy, remark):
        self.calls.append(dict(time=self.clock.time(), code=code, side=side, volume=volume,
                               price_type=price_type, price=price, remark=remark))
        order = NS(order_id=len(self.orders) + 1, stock_code=code, order_type=side,
                   order_volume=volume, traded_volume=0, order_status=50, traded_price=0.0,
                   status_msg='', order_time=self.clock.time(), price_type=price_type,
                   price=price, strategy_name=strategy, order_remark=remark)
        self.orders.append(order)
        if self.scenario == 'rejected_then_retry' and not self.rejected:
            self.rejected = True
            order.order_status = 57
            order.status_msg = '150906130 insufficient funds (simulated explicit broker rejection)'
        if self.scenario == 'lost_ack_recovery' and not self.crashed:
            self.crashed = True
            raise RuntimeError('Simulated disconnect after broker acceptance')
        return 0 if self.scenario == 'zero_accepted' else order.order_id

    def cancel_order_stock(self, account, order_id):
        order = self.query_stock_order(account, order_id)
        self.cancels.append(dict(time=self.clock.time(), order_id=order_id))
        order.order_status = 53 if order.traded_volume else 54
        return 0


def run_case(name, day):
    from trading.app_executor import AppBroker, AppTrader, FileExecutionService
    from trading.bridge_wire import atomic_json, open_database
    from trading import execution_engine

    clock = Clock(day)
    execution_engine.time = clock
    root = Path(tempfile.mkdtemp(prefix='wbr-fake-broker-'))
    directory = root / 'file'
    directory.mkdir()
    if name == 'keep_small_kcb_buy':
        holdings, cash, targets = {'688092.SH': 200}, 10000, {'688092.SH': 400}
    elif name == 'keep_odd_main_buy':
        holdings, cash, targets = {'600001.SH': 100}, 10000, {'600001.SH': 400}
    elif name == 'keep_small_kcb_sell':
        holdings, cash, targets = {'688092.SH': 1000}, 10000, {'688092.SH': 600}
    elif name in ('sell_then_cash_wait', 'preopen_sell_before_buy'):
        holdings, cash = {'600002.SH': 200}, 100
        targets = {'600002.SH': 0, '600001.SH': 200}
        if name == 'preopen_sell_before_buy':
            holdings['600001.SH'] = 100
            cash = 10000
            targets['600001.SH'] = 300
    else:
        holdings, cash = {'600001.SH': 100}, 10000
        targets = {'600001.SH': 300}
    sdk = FakeSDK(clock, holdings, cash, name)
    database = open_database(root / 'simulation.sqlite3')
    config = dict(account='SIMULATED', root=str(root), live_enabled=True, execution_mode='file', simulation=True,
                  strategy_name='FlashMan', trading_dates=[day.isoformat()], price_types=[11, 44])
    # Only this simulation injects a fake owner check and FakeSDK. Production
    # main verifies the real brokerage parent and holds the single-instance lock.
    bridge = AppBroker(config, sdk, NS(account_id='SIMULATED'), database,
                       clock=clock.time, owner_check=lambda pid: None)
    target = dict(version=1, id='012345abcdef', account='SIMULATED', trade_date=day.isoformat(),
                  executor_protocol=FILE_EXECUTOR_PROTOCOL,simulation=True,
                  strategy_id='SIMULATED',model_sha256='0'*64,max_positions=40,
                  generated_at=clock.time() - 1, expires=clock.time() + 120,
                  targets={code: dict(shares=shares, open_price=sdk.price(code),
                                      freeze_price=sdk.price(code) * 1.1) for code, shares in targets.items()},
                  buy_priority=[c for c in targets if targets[c] > holdings.get(c, 0)])
    target_path = directory / ('targets_' + day.isoformat() + '.json')
    atomic_json(target_path, target)
    service = FileExecutionService(AppTrader(bridge), directory)
    service.engine.SETTLE_WAIT_SEC = 0
    service.engine.BUY_MONITOR_DEADLINE_SEC = 180
    recovered = False
    try:
        result = service.tick(clock.now())
    except RuntimeError:
        if name != 'lost_ack_recovery':
            raise
        target_path.unlink()
        clock.sleep(130)  # Expired AND removed source; recover frozen journal.
        service = FileExecutionService(AppTrader(bridge), directory)
        service.engine.SETTLE_WAIT_SEC = 0
        service.engine.BUY_MONITOR_DEADLINE_SEC = 180
        recovered = True
        result = service.tick(clock.now())
    assert result['stage'] == 'completed', (name, result, sdk.calls)
    assert all(sdk.holdings.get(code, 0) == quantity for code, quantity in targets.items())
    assert all(datetime.datetime.fromtimestamp(c['time']).time() >= datetime.time(9, 30) for c in sdk.calls)
    calls_before = len(sdk.calls)
    service.tick(clock.now())
    assert len(sdk.calls) == calls_before, 'Repeated file read submitted another order'
    daily_file = directory / 'daily' / ('executor_' + day.isoformat() + '.json')
    daily = json.loads(daily_file.read_text(encoding='utf-8'))
    assert daily['simulated'] and daily['state'] == 'completed' and daily['evidence_verified']
    if name == 'partial_reprice':
        assert len(sdk.calls) == 2 and sdk.cancels
        assert sdk.calls[0]['price_type'] == 11 and sdk.calls[1]['price_type'] == 44
    elif name == 'sell_then_cash_wait':
        assert [call['side'] for call in sdk.calls] == [24, 23]
        assert sdk.calls[1]['time'] >= sdk.calls[0]['time'] + 2
    elif name == 'preopen_sell_before_buy':
        assert [call['side'] for call in sdk.calls] == [24, 23]
    elif name in ('lost_ack_recovery', 'zero_accepted'):
        assert len(sdk.calls) == 1
        assert recovered if name == 'lost_ack_recovery' else True
    elif name == 'rejected_then_retry':
        assert len(sdk.calls) == 2
    elif name in ('keep_small_kcb_buy', 'keep_odd_main_buy', 'keep_small_kcb_sell'):
        assert len(sdk.calls) == 1 and not sdk.cancels
    result = dict(name=name, passed=True, fake_submissions=sdk.calls, fake_cancellations=sdk.cancels,
                  daily_record_file=str(daily_file),
                  final_positions=sdk.holdings, journal_requests=len(service.journal.state['requests']),
                  rpc_requests=database.execute('SELECT count(*) FROM requests').fetchone()[0])
    database.close()
    return result


def main(report_path):
    sys.meta_path.insert(0, DenyLiveBroker())
    socket.socket = deny_network
    report = dict(python=sys.version, executable=sys.executable, pid=os.getpid(),
                  simulated=True, real_broker_connections=0, real_orders_or_cancels=0)
    try:
        day = datetime.date.today()
        report['cases'] = [run_case(name, day) for name in (
            'partial_reprice', 'sell_then_cash_wait', 'rejected_then_retry', 'lost_ack_recovery',
            'preopen_sell_before_buy', 'keep_small_kcb_buy', 'keep_odd_main_buy', 'keep_small_kcb_sell', 'zero_accepted')]
        report['passed'] = True
    except BaseException:
        report['passed'] = False
        report['error'] = traceback.format_exc()
    Path(report_path).write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding='utf-8')


if __name__ == '__main__':
    report_path = sys.argv[1]
    logging.basicConfig(filename=report_path + '.log', level=logging.INFO)
    main(report_path)
