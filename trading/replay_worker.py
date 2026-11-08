"""Historical client-side replay, Python 3.6. Real broker/network are forbidden."""
from collections import Counter
import datetime
import hashlib
import html
import json
import logging
import os
from pathlib import Path
import socket
import sys
import tempfile
import traceback
from types import SimpleNamespace as NS

from trading.app_simulation_worker import Clock, DenyLiveBroker, FakeSDK, deny_network
from trading.holding_log import comparison_lines, holding_comparison


class ReplaySDK(FakeSDK):
    def __init__(self, clock, fixture):
        super().__init__(clock, {p['stock_code']: p['volume'] for p in fixture['positions']},
                         fixture['asset']['cash'], 'historical_replay')
        self.price_map = {p['stock_code']: p['market_value'] / p['volume'] for p in fixture['positions']}
        self.price_map.update({c: r['open_price'] for c, r in fixture['document']['targets'].items()})
        self.account_id = fixture['account']
        self.simulated_trades = []
        self.delayed_code = next((r['code'] for r in reversed(fixture['expected_orders'])
                                  if r['direction'] == 'BUY'), None)

    def price(self, code):
        return self.price_map[code]

    def update(self):
        for order in self.orders:
            if order.order_status in (53, 54, 56, 57):
                continue
            # Demonstrate the real 09:31 transition on one planned buy. This
            # is an explicit matching scenario, not a claim about Nov 2 fills.
            if order.stock_code == self.delayed_code and order.price_type == 11:
                continue
            if self.clock.time() - order.order_time >= 2:
                self.fill(order, order.order_volume - order.traded_volume)

    def fill(self, order, quantity):
        if quantity <= 0:
            return
        super().fill(order, quantity)
        self.simulated_trades.append(NS(account_id=self.account_id,
            traded_id='SIM%06d' % (len(self.simulated_trades) + 1), order_id=order.order_id,
            stock_code=order.stock_code, order_type=order.order_type,
            traded_volume=quantity, traded_price=self.price(order.stock_code),
            traded_amount=quantity * self.price(order.stock_code), traded_time=self.clock.time(),
            strategy_name=order.strategy_name, order_remark=order.order_remark))

    def query_stock_trades(self, account):
        self.update()
        return self.simulated_trades

    def query_stock_asset(self, account):
        result = super().query_stock_asset(account)
        result.account_id = self.account_id
        return result

    def query_stock_positions(self, account):
        result = super().query_stock_positions(account)
        for position in result:
            position.account_id = self.account_id
        return result


class ReplayLog:
    def __init__(self, path, clock):
        self.path, self.clock = path, clock
        self.stream = path.open('w', encoding='utf-8')

    def write(self, message):
        line = '[模拟 %s] %s' % (self.clock.now().strftime('%Y-%m-%d %H:%M:%S'), message)
        self.stream.write(line + '\n')
        self.stream.flush()
        print(line, flush=True)


class ReplayHandler(logging.Handler):
    def __init__(self, writer):
        super().__init__()
        self.writer = writer

    def emit(self, record):
        self.writer.write(record.getMessage())


def run_replay(fixture_path):
    sys.meta_path.insert(0, DenyLiveBroker())
    socket.socket = deny_network
    from trading.app_executor import AppBroker, AppTrader, FileExecutionService
    from trading.bridge_wire import atomic_json, encode, open_database
    from trading import execution_engine

    fixture_path = Path(fixture_path)
    fixture = json.loads(fixture_path.read_text(encoding='utf-8'))
    if fixture['simulation_only'] is not True:
        raise RuntimeError('Only a verified simulation fixture may be replayed')
    day = datetime.datetime.strptime(fixture['trade_date'], '%Y-%m-%d').date()
    start = datetime.datetime.fromtimestamp(fixture['document']['generated_at'])
    clock = Clock(day, start.hour, start.minute, start.second)
    if clock.time() < fixture['document']['generated_at']:
        clock.sleep(1)
    execution_engine.time = clock
    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = fixture_path.parent / ('run_' + run_id)
    output.mkdir()
    writer = ReplayLog(output / 'replay.log', clock)
    logger = logging.getLogger('wbr.execution')
    logger.handlers = [ReplayHandler(writer)]
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    report = dict(simulated=True, replay_date=day.isoformat(), actual_run_time=datetime.datetime.now().isoformat(),
        python=sys.version, executable=sys.executable, pid=os.getpid(), parent_pid=os.getppid(),
        stdout_encoding=getattr(sys.stdout, 'encoding', None), source_fixture=str(fixture_path),
        source_fixture_sha256=hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        real_broker_connections=0, real_orders_or_cancels=0, passed=False)
    database = None
    try:
        writer.write('历史执行回放开始；真实账户连接/报撤单为零；未修改系统时间')
        writer.write('读取历史账户：%s，只数=%d，现金=%.2f，原采集时间=%s' % (
            fixture['account'], len(fixture['positions']), fixture['asset']['cash'],
            datetime.datetime.fromtimestamp(fixture['account_captured_at']).isoformat()))
        writer.write('策略来源：11月2日已保存的原始计划；本程序不计算因子或重新选股')
        writer.write('撮合假设：按计划开盘价成交、无手续费/滑点；最后一笔计划买入的限价单等待09:31后换对手价')
        rows = holding_comparison(fixture['document'], fixture['positions'], fixture['names'])
        for line in comparison_lines(rows):
            writer.write(line)
        root = Path(tempfile.mkdtemp(prefix='wbr-historical-fake-'))
        directory = root / 'file'
        directory.mkdir()
        sdk = ReplaySDK(clock, fixture)
        database = open_database(root / 'simulation.sqlite3')
        config = dict(account=fixture['account'], root=str(root), live_enabled=True, execution_mode='file', simulation=True,
                      strategy_name='WBR_HISTORICAL_SIM', trading_dates=[day.isoformat()], price_types=[11, 44])
        bridge = AppBroker(config, sdk, NS(account_id=fixture['account']), database,
                           clock=clock.time, owner_check=lambda pid: None)
        atomic_json(directory / ('targets_' + day.isoformat() + '.json'), fixture['document'])
        service = FileExecutionService(AppTrader(bridge), directory, now_fn=clock.now)
        service.engine.BUY_MONITOR_DEADLINE_SEC = 450
        service.engine.SETTLE_WAIT_SEC = 0
        writer.write('已接收目标；持续轮询模拟时钟，09:30以前禁止发单')
        result = service.tick()
        expected = Counter({(r['code'], r['direction']): r['shares'] for r in fixture['expected_orders']})
        filled = Counter()
        for order in sdk.orders:
            filled[(order.stock_code, 'BUY' if order.order_type == 23 else 'SELL')] += order.traded_volume
        assert result['stage'] == 'completed', result
        assert filled == expected, (filled, expected)
        assert all(datetime.datetime.fromtimestamp(c['time']).time() >= datetime.time(9, 30) for c in sdk.calls)
        initial_sides = [call['side'] for call in sdk.calls[:len(expected)]]
        if any(side == 24 for side in initial_sides) and any(side == 23 for side in initial_sides):
            assert initial_sides == sorted(initial_sides, reverse=True), 'Initial sells must precede buys'
        if sdk.delayed_code is not None:
            assert sdk.cancels, 'The 09:31 cancellation/peer transition was not exercised'
        before = len(sdk.calls)
        service.tick()
        assert len(sdk.calls) == before, 'Repeated target read generated duplicate orders'
        writer.write('模拟成交汇总：原计划5笔方向/数量全部覆盖；委托%d笔（含换价重挂），撤单%d次' %
                     (len(sdk.calls), len(sdk.cancels)))
        final = []
        for row in rows:
            item = dict(row, final=sdk.holdings.get(row['code'], 0))
            final.append(item)
            writer.write('完成核对 %s：盘前=%d，目标=%d，模拟后=%d' % (
                row['code'], row['current'], row['target'], item['final']))
        assert all(r['final'] == r['target'] for r in final)
        report.update(passed=True, comparison=final, simulated_orders=encode(sdk.orders),
                      simulated_trades=encode(sdk.simulated_trades), cancellations=sdk.cancels,
                      first_order_time=(datetime.datetime.fromtimestamp(sdk.calls[0]['time']).isoformat()
                                        if sdk.calls else None),
                      completed_at=clock.now().isoformat(), rpc_requests=database.execute(
                          'SELECT count(*) FROM requests').fetchone()[0])
        writer.write('重复读取目标：新增委托0。结果仅为模拟，不代表1102真实成交或真实报单权限。')
        cells = ''.join('<tr><td>%s</td><td>%s</td><td>%d</td><td>%d</td><td>%s</td><td>%d</td></tr>' %
                        (html.escape(r['code']), html.escape(r['name']), r['current'], r['target'],
                         html.escape(r['action']), r['final']) for r in final)
        page = '<!doctype html><meta charset="utf-8"><title>1102持仓执行模拟</title>'
        page += '<style>body{font:16px sans-serif;margin:30px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:8px}</style>'
        page += '<h1>2026-11-02 历史执行模拟</h1><p>真实盘前快照 + 原始计划；模拟撮合；真实报撤单为零。</p>'
        page += '<table><tr><th>代码</th><th>名称</th><th>账户持仓</th><th>目标持仓</th><th>多退少补</th><th>模拟后</th></tr>' + cells + '</table>'
        (output / 'comparison.html').write_text(page, encoding='utf-8')
    except BaseException:
        report['error'] = traceback.format_exc()
        writer.write('回放失败：' + report['error'])
    finally:
        if database is not None:
            database.close()
        atomic_json(output / 'report.json', report)
        atomic_json(fixture_path.parent / 'latest.json', dict(report=str(output / 'report.json'),
            log=str(output / 'replay.log'), passed=report['passed'], actual_run_time=report['actual_run_time']))
        writer.write('回放证据：' + str(output / 'report.json'))
        writer.stream.close()
    return report


if __name__ == '__main__':
    fixture = os.environ.get('WBR_REPLAY_INPUT') or sys.argv[1]
    run_replay(fixture)
