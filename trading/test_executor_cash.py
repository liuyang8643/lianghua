import time
from datetime import date, datetime, time as dt_time
from types import SimpleNamespace

from xtquant import xtconstant

from trading.executor import OrderMonitor, RebalanceExecutor


class _FakeTrader:
    """可控 trader:order 默认全成,可用 on_order 钩子改写订单结局;query_asset 返回可用现金。"""

    def __init__(self, cash: float = 1e9):
        self.orders: dict[int, SimpleNamespace] = {}
        self.cash = cash
        self._next = 1000
        self.on_order = None  # (code, shares, oid) -> SimpleNamespace | None
        self.submitted = []  # (code, shares, oid, price)
        self.cancelled = []
        self.can_use = {}

    def query_asset(self):
        return SimpleNamespace(cash=self.cash)

    def order(self, order_type, code, shares, price, order_remark=''):
        oid = self._next
        self._next += 1
        self.submitted.append((code, shares, oid, price))
        o = SimpleNamespace(stock_code=code, order_type=order_type,
                            order_status=xtconstant.ORDER_SUCCEEDED,
                            traded_volume=shares, order_volume=shares,
                            traded_price=0.0, status_msg='')
        if self.on_order:
            o = self.on_order(code, shares, oid) or o
        self.orders[oid] = o
        return oid

    def query_order(self, oid):
        return self.orders.get(oid)

    def cancel_order(self, oid):
        self.cancelled.append(oid)
        o = self.orders.get(oid)
        if o:
            o.order_status = (
                xtconstant.ORDER_PART_CANCEL
                if int(getattr(o, 'traded_volume', 0) or 0) > 0
                else xtconstant.ORDER_CANCELED
            )

    def query_stock_position(self, code):
        return SimpleNamespace(
            stock_code=code, can_use_volume=self.can_use.get(code, 0))


def _make_order_monitor(executor, allocations=None, prices=None, limit_prices=None,
                         sell_orders=None):
    # 普通单元测试固定在 09:31 前；时间线测试直接构造 monitor。
    executor.OPEN_LIMIT_DEADLINE_SEC = 24 * 60 * 60
    om = OrderMonitor(
        executor,
        sell_orders=sell_orders or [],
        buy_allocations=allocations or {},
        buy_n_stocks=list(allocations.keys()) if allocations else [],
        prices=prices or {},
        signal_date=date.today(),
        trade_date=date.today(),
        limit_prices=limit_prices,
    )
    om.record_action = lambda *a, **k: None
    return om


# ── 可用资金口径 ─────────────────────────────────────────

def test_available_buy_cash_uses_xtasset_cash():
    """回归 2026-06-11:current_balance/fetch_balance 是上日结存/可取金额口径,
    盘中不反映卖出回款 → 必须用 XtAsset.cash(可用金额),且无视这两个字段。"""
    asset = SimpleNamespace(cash=100_000.0, current_balance=12_345.0,
                            fetch_balance=12_000.0, frozen_cash=5_000.0)
    assert RebalanceExecutor._available_buy_cash(asset) == 100_000.0


def test_available_buy_cash_handles_missing_asset():
    assert RebalanceExecutor._available_buy_cash(None) == 0.0
    assert RebalanceExecutor._available_buy_cash(SimpleNamespace(cash=None)) == 0.0


# ── 涨停价 / 板块比例 ─────────────────────────────────────

def test_board_limit_ratio_and_limit_up_price():
    from utils.stock.info import board_limit_ratio, limit_up_price
    assert board_limit_ratio('688296.SH') == 0.20  # 科创板
    assert board_limit_ratio('300001.SZ') == 0.20  # 创业板
    assert board_limit_ratio('600000.SH') == 0.10  # 主板
    assert board_limit_ratio('830799.BJ') == 0.30  # 北交所
    assert abs(limit_up_price('688296.SH', 14.36) - 14.36 * 1.20) < 1e-9
    assert abs(limit_up_price('600000.SH', 10.0) - 11.0) < 1e-9
    assert limit_up_price('600000.SH', 0) == 0.0


# ── 买入单位成本(涨停价冻结口径) ──────────────────────────

def test_unit_cost_uses_limit_price_when_present():
    ex = RebalanceExecutor(_FakeTrader())
    code = '688296.SH'
    om = _make_order_monitor(ex, {code: 200}, {code: 14.3}, limit_prices={code: 17.23})
    expected = 17.23 * (1 + ex.BUY_FEE_RATE)
    assert abs(om._unit_cost(code) - expected) < 1e-9
    # 涨停价口径显著高于开盘价口径,避免低估资金占用
    assert om._unit_cost(code) > 14.3 * (1 + ex.BUY_FEE_RATE)


def test_unit_cost_falls_back_to_open_when_no_limit():
    ex = RebalanceExecutor(_FakeTrader())
    code = '600000.SH'
    om = _make_order_monitor(ex, {code: 100}, {code: 10.0})
    expected = 10.0 * (1 + ex.BUY_FEE_RATE)
    assert abs(om._unit_cost(code) - expected) < 1e-9


# ── 卖单在途感知 / 缺口计算 ────────────────────────────────

def _inject_inflight_sell(trader, om, code='600000.SH', oid=9000, volume=100):
    """给 OrderMonitor 注入一笔「未终态」卖单。"""
    trader.orders[oid] = SimpleNamespace(
        stock_code=code, order_status=xtconstant.ORDER_REPORTED,
        order_volume=volume, traded_volume=0, status_msg='')
    om.submitted.append({'code': code, 'order_id': oid, 'shares': volume,
                         'submitted_at': 0, 'order_type': 'SELL'})
    om.orders_by_code.setdefault(code, []).append(oid)
    om.sell_targets[code] = volume


def test_sell_submit_registers_and_tracks_pending_order():
    """卖单提交后登记进 submitted，统一在途状态实时反映订单终态。"""
    trader = _FakeTrader()
    trader.on_order = lambda code, shares, oid: SimpleNamespace(
        stock_code=code, order_status=xtconstant.ORDER_REPORTED,
        traded_volume=0, order_volume=shares, traded_price=0.0,
        order_type=xtconstant.STOCK_SELL, status_msg='')
    ex = RebalanceExecutor(trader)
    om = _make_order_monitor(ex, sell_orders=[('600000.SH', 700)])
    om._submit('SELL', '600000.SH', 700)
    assert len(om.submitted) == 1
    assert om._has_pending_orders() is True
    assert om._sell_remaining('600000.SH') == 0

    oid = om.submitted[0]['order_id']
    trader.orders[oid].order_status = xtconstant.ORDER_JUNK
    assert om._has_pending_orders() is False
    assert om._sell_remaining('600000.SH') == 700


def test_pending_orders_reflect_terminal_state():
    """回归 2026-06-11:在途与否必须实时查订单终态,不能依赖标志位。"""
    trader = _FakeTrader()
    ex = RebalanceExecutor(trader)
    om = _make_order_monitor(ex, {'600000.SH': 100}, {'600000.SH': 10.0})
    assert om._has_pending_orders() is False

    _inject_inflight_sell(trader, om, '600000.SH', 9000, 100)
    assert om._has_pending_orders() is True

    trader.orders[9000].order_status = xtconstant.ORDER_SUCCEEDED
    assert om._has_pending_orders() is False


def test_transient_query_none_keeps_known_order_inflight():
    """订单曾查到后短暂查询失败时，保守视为仍在途，不能重复下单。"""
    trader = _FakeTrader(cash=1e7)
    trader.on_order = lambda code, shares, oid: SimpleNamespace(
        stock_code=code, order_type=xtconstant.STOCK_BUY,
        order_status=xtconstant.ORDER_REPORTED,
        traded_volume=0, order_volume=shares, traded_price=0.0, status_msg='')
    ex = RebalanceExecutor(trader)
    code = '600000.SH'
    om = _make_order_monitor(ex, {code: 100}, {code: 10.0})
    assert om._submit_affordable_buys() is True
    assert om._has_open_order(code) is True

    trader.orders.pop(1000)
    assert om._remaining(code) == 0
    assert om._submit_affordable_buys() is False
    assert len(trader.submitted) == 1

    trader.orders[1000] = SimpleNamespace(
        stock_code=code, order_type=xtconstant.STOCK_BUY,
        order_status=xtconstant.ORDER_CANCELED,
        traded_volume=0, order_volume=100, traded_price=0.0, status_msg='')
    assert om._remaining(code) == 100
    trader.orders.pop(1000)
    assert om._remaining(code) == 100


def test_cached_reject_still_sets_backoff_when_query_turns_none():
    """已缓存的终态废单即使随后查不到，也必须先执行退避副作用。"""
    trader = _FakeTrader(cash=1e7)
    trader.on_order = lambda code, shares, oid: SimpleNamespace(
        stock_code=code, order_type=xtconstant.STOCK_BUY,
        order_status=xtconstant.ORDER_REPORTED,
        traded_volume=0, order_volume=shares, traded_price=0.0, status_msg='')
    ex = RebalanceExecutor(trader)
    code = '600000.SH'
    om = _make_order_monitor(ex, {code: 100}, {code: 10.0})
    om._submit_affordable_buys()

    trader.orders[1000].order_status = xtconstant.ORDER_JUNK
    trader.orders[1000].status_msg = 'price error'
    assert om._remaining(code) == 100  # 缓存终态
    trader.orders.pop(1000)
    om._handle_terminal_orders()

    assert om.retry_after[code] > time.time()
    assert om._submit_affordable_buys() is False
    assert len(trader.submitted) == 1


def test_terminal_status_seen_after_handler_does_not_duplicate_order():
    """同一轮稍后才查到终态时，先等终态处理，不能抢先补单。"""

    class _StatusJumpTrader(_FakeTrader):
        def __init__(self):
            super().__init__(cash=1e7)
            self.query_count = 0

        def query_order(self, oid):
            order = super().query_order(oid)
            self.query_count += 1
            if self.query_count >= 2:
                order.order_status = xtconstant.ORDER_JUNK
                order.status_msg = 'price error'
            return order

    trader = _StatusJumpTrader()
    trader.on_order = lambda code, shares, oid: SimpleNamespace(
        stock_code=code, order_type=xtconstant.STOCK_BUY,
        order_status=xtconstant.ORDER_REPORTED,
        traded_volume=0, order_volume=shares, traded_price=0.0, status_msg='')
    ex = RebalanceExecutor(trader)
    code = '600000.SH'
    om = _make_order_monitor(ex, {code: 100}, {code: 10.0})
    assert om._submit_affordable_buys() is True

    om._handle_terminal_orders()  # 第一次仍是已报
    assert om._submit_affordable_buys() is False  # 此时跳为废单
    assert len(trader.submitted) == 1

    om._handle_terminal_orders()
    assert om.retry_after[code] > time.time()


def test_remaining_counts_filled_and_inflight():
    ex = RebalanceExecutor(_FakeTrader())
    code = '600000.SH'
    om = _make_order_monitor(ex, {code: 1000}, {code: 10.0})
    om.trader.orders[1] = SimpleNamespace(
        order_status=xtconstant.ORDER_SUCCEEDED, traded_volume=300, order_volume=300)
    om.trader.orders[2] = SimpleNamespace(
        order_status=xtconstant.ORDER_REPORTED, traded_volume=0, order_volume=200)
    om.orders_by_code[code] = [1, 2]
    assert om._remaining(code) == 1000 - 300 - 200


# ── 持久挂单核心 ─────────────────────────────────────────

def test_submit_affordable_buys_places_limit_order_once():
    """09:31 前现金够时直接挂 open[T] 限价单。"""
    ex = RebalanceExecutor(_FakeTrader(cash=1e7))
    code = '600000.SH'
    om = _make_order_monitor(ex, {code: 1000}, {code: 10.0}, limit_prices={code: 11.0})
    assert om._submit_affordable_buys() is True
    assert om._remaining(code) == 0
    assert len(om.submitted) == 1
    assert ex.trader.submitted[0][3] == 10.0


def test_reject_retries_same_size_later_without_blocking_or_shrinking():
    """非资金类废单不减手、不熔断；冷却后仍按原 open[T] 和原缺口重试。"""
    trader = _FakeTrader(cash=1e7)
    trader.on_order = lambda code, shares, oid: SimpleNamespace(
        order_status=xtconstant.ORDER_JUNK, traded_volume=0,
        order_volume=shares, traded_price=0.0, status_msg='订单价格超出范围')
    ex = RebalanceExecutor(trader)
    code = '600000.SH'
    om = _make_order_monitor(ex, {code: 1000}, {code: 10.0}, limit_prices={code: 11.0})
    assert om._submit_affordable_buys() is True
    om._handle_terminal_orders()
    assert om._remaining(code) == 1000
    assert om.retry_after[code] > 0
    om.retry_after[code] = 0
    assert om._submit_affordable_buys() is True
    assert [x[1] for x in trader.submitted] == [1000, 1000]


def test_run_keeps_monitoring_reported_order_and_retries_late_reject():
    """回归 2026-07-17:已报订单数分钟后才废单，主循环不能提前退出。"""
    class _LateRejectTrader(_FakeTrader):
        def order(self, order_type, code, shares, price, order_remark=''):
            oid = super().order(order_type, code, shares, price, order_remark)
            o = self.orders[oid]
            o.order_type = order_type
            if len(self.submitted) == 1:
                o.order_status = xtconstant.ORDER_REPORTED
                o.traded_volume = 0
                o.reject_at = time.time() + 0.02
            return oid

        def query_order(self, oid):
            o = super().query_order(oid)
            if (o and getattr(o, 'reject_at', None)
                    and time.time() >= o.reject_at):
                o.order_status = xtconstant.ORDER_JUNK
                o.status_msg = '订单价格超出范围'
            return o

    trader = _LateRejectTrader(cash=1e7)
    ex = RebalanceExecutor(trader)
    ex.BUY_MONITOR_DEADLINE_SEC = 0.2
    ex.ORDER_REJECT_RETRY_SEC = 0.01
    ex.MONITOR_POLL_SEC = 0.001
    code = '688616.SH'
    om = _make_order_monitor(
        ex, {code: 3222}, {code: 10.23}, limit_prices={code: 12.25})

    om.run()

    assert [(shares, price) for _, shares, _, price in trader.submitted] == [
        (3222, 10.23), (3222, 10.23),
    ]
    assert om._remaining(code) == 0


def test_underfunded_reject_waits_for_cash_without_blocking():
    """资金不足废单只退避等 cash 更新,不熔断也不减手。"""
    trader = _FakeTrader(cash=1e7)
    trader.on_order = lambda code, shares, oid: SimpleNamespace(
        order_status=xtconstant.ORDER_JUNK, traded_volume=0,
        order_volume=shares, traded_price=0.0, status_msg='资金可用数不足,尚需3440')
    ex = RebalanceExecutor(trader)
    code = '688296.SH'
    om = _make_order_monitor(ex, {code: 600}, {code: 10.0}, limit_prices={code: 11.0})
    assert om._submit_affordable_buys() is True
    om._handle_terminal_orders()
    assert om._remaining(code) == 600
    assert om.retry_after[code] > 0


def test_is_insufficient_funds_recognizes_qmt_numeric_code():
    """回归 2026-06-02:query_order().status_msg 只给数字码 -150906130|1|-150906130,
    不含中文「资金可用数不足」。必须靠数字码识别,否则资金废单被误判为硬废单。"""
    from trading.executor import _is_insufficient_funds
    assert _is_insufficient_funds('-150906130|1|-150906130') is True   # QMT 买入资金不足码
    assert _is_insufficient_funds('资金可用数不足,尚需3440') is True
    assert _is_insufficient_funds('insufficient buying power') is True
    assert _is_insufficient_funds('无对手方价格,废单') is False         # 真·硬废单
    assert _is_insufficient_funds('') is False


def test_numeric_code_reject_uses_underfunded_retry():
    """资金不足废单只给数字码(无中文)时,也按资金退避处理。"""
    trader = _FakeTrader(cash=1e7)
    trader.on_order = lambda code, shares, oid: SimpleNamespace(
        order_status=xtconstant.ORDER_JUNK, traded_volume=0,
        order_volume=shares, traded_price=0.0, status_msg='-150906130|1|-150906130')
    ex = RebalanceExecutor(trader)
    code = '688466.SH'  # 科创 min_lot=200
    om = _make_order_monitor(ex, {code: 600}, {code: 10.0}, limit_prices={code: 11.0})
    om._submit_affordable_buys()
    om._handle_terminal_orders()
    assert om._remaining(code) == 600
    assert om.retry_after[code] > 0


def test_cash_gate_waits_until_cash_is_available():
    """现金不够时不下废单；QMT cash 到位后才挂后续买单。"""
    trader = _FakeTrader(cash=1500.0)
    ex = RebalanceExecutor(trader)
    code = '688296.SH'  # 科创最少 200 股,按涨停价 11 估需 2200+
    om = _make_order_monitor(ex, {code: 2000}, {code: 10.0}, limit_prices={code: 11.0})
    assert om._submit_affordable_buys() is False
    assert trader.submitted == []
    trader.cash = 11.0 * (1 + ex.BUY_FEE_RATE) * 250 + 0.01
    assert om._submit_affordable_buys() is True
    assert trader.submitted[0][1] == 250


# ── 卖单撤单重挂(等股份释放 + 可用量封顶) ──────────────────

class _FakeSellTrader:
    """卖单重挂场景:order 全成;cancel 把原委托置 CANCELED 并(可选)释放未成交股份回可用。"""

    def __init__(self, release_on_cancel=True):
        self.orders: dict[int, SimpleNamespace] = {}
        self.can_use: dict[str, int] = {}
        self._next = 5000
        self.sell_submits = []
        self.release_on_cancel = release_on_cancel

    def query_order(self, oid):
        return self.orders.get(oid)

    def query_stock_position(self, code):
        return SimpleNamespace(stock_code=code, can_use_volume=self.can_use.get(code, 0))

    def cancel_order(self, oid):
        o = self.orders.get(oid)
        if not o:
            return
        o.order_status = xtconstant.ORDER_CANCELED
        if self.release_on_cancel:
            self.can_use[o.stock_code] = self.can_use.get(o.stock_code, 0) + (
                o.order_volume - o.traded_volume)

    def order(self, order_type, code, shares, price, order_remark=''):
        oid = self._next
        self._next += 1
        self.sell_submits.append((code, shares, oid))
        self.orders[oid] = SimpleNamespace(
            stock_code=code, order_status=xtconstant.ORDER_SUCCEEDED,
            order_volume=shares, traded_volume=shares, status_msg='')
        self.can_use[code] = max(0, self.can_use.get(code, 0) - shares)
        return oid


def _stuck_order_monitor(ex, code, volume):
    """创建含「卡住」卖单的 OrderMonitor，模拟在途卖单场景。"""
    om = _make_order_monitor(ex, sell_orders=[(code, volume)])
    ex.trader.orders[5000] = SimpleNamespace(
        stock_code=code, order_status=xtconstant.ORDER_REPORTED,
        order_volume=volume, traded_volume=0, order_type=xtconstant.STOCK_SELL,
        status_msg='')
    ex.trader.can_use[code] = 0
    sub = {'code': code, 'order_id': 5000, 'shares': volume,
           'submitted_at': 0, 'order_type': 'SELL'}
    om.submitted.append(sub)
    om.orders_by_code.setdefault(code, []).append(5000)
    om.sell_targets[code] = volume
    return om, sub


def test_execute_does_not_cancel_peer_order_before_ttl(monkeypatch):
    """对手价单未满 30 分钟时不撤单。"""
    class _PendingTrader(_FakeTrader):
        def __init__(self):
            super().__init__(cash=1e7)
            self.cancel_count = 0
        def order(self, order_type, code, shares, price, order_remark=''):
            oid = super().order(order_type, code, shares, price, order_remark)
            self.orders[oid].order_status = xtconstant.ORDER_REPORTED
            self.orders[oid].traded_volume = 0
            return oid
        def cancel_order(self, oid):
            self.cancel_count += 1

    clock = _FakeClock(datetime.combine(date.today(), dt_time(10, 0)))
    monkeypatch.setattr('trading.executor.time', clock)
    monkeypatch.setattr('trading.executor.is_current_trading', lambda _: True)
    trader = _PendingTrader()
    ex = RebalanceExecutor(trader)
    ex.OPEN_LIMIT_DEADLINE_SEC = 0
    ex.BUY_MONITOR_DEADLINE_SEC = 0.1
    ex.SETTLE_WAIT_SEC = 0
    pending = {
        'signal_date': date.today(), 'trade_date': date.today(),
        'sell_orders': [('600001.SH', 100)],
        'buy_allocations': {'600002.SH': 100},
        'buy_n_stocks': ['600002.SH'],
        'prices': {'600001.SH': 10.0, '600002.SH': 10.0},
        'limit_prices': {'600002.SH': 11.0},
    }
    ex.execute(pending)
    assert len(trader.submitted) == 2
    assert trader.cancel_count == 0


class _FakeClock:
    def __init__(self, current: datetime):
        self.current = current.timestamp()

    def time(self):
        return self.current

    def sleep(self, seconds):
        self.current += seconds


def test_open_order_switches_at_0931_then_reposts_peer_every_30m(monkeypatch):
    """首单用 open；09:31 撤余改对手价，此后每 30 个连续交易分钟刷新。"""
    start = datetime.combine(date.today(), dt_time(9, 25, 10))
    clock = _FakeClock(start)
    monkeypatch.setattr('trading.executor.time', clock)
    monkeypatch.setattr('trading.executor.is_current_trading', lambda _: True)

    trader = _FakeTrader(cash=1e7)

    def pending_order(code, shares, oid):
        first = oid == 1000
        return SimpleNamespace(
            stock_code=code, order_type=xtconstant.STOCK_BUY,
            order_status=(xtconstant.ORDER_PART_SUCC if first
                          else xtconstant.ORDER_REPORTED),
            traded_volume=300 if first else 0, order_volume=shares,
            traded_price=10.0 if first else 0.0, status_msg='')

    trader.on_order = pending_order
    ex = RebalanceExecutor(trader)
    ex.BUY_MONITOR_DEADLINE_SEC = (
        datetime.combine(date.today(), dt_time(10, 1, 20)).timestamp()
        - start.timestamp()
    )
    ex.MONITOR_POLL_SEC = 10
    code = '600000.SH'
    om = OrderMonitor(
        ex,
        sell_orders=[],
        buy_allocations={code: 1000},
        buy_n_stocks=[code],
        prices={code: 10.0},
        signal_date=date.today(),
        trade_date=date.today(),
        limit_prices={code: 11.0},
    )
    om.record_action = lambda *a, **k: None

    om.run()

    assert [(shares, price) for _, shares, _, price in trader.submitted] == [
        (1000, 10.0),
        (700, None),
        (700, None),
    ]
    assert trader.cancelled == [1000, 1001]
    assert trader.orders[1000].order_status == xtconstant.ORDER_PART_CANCEL
    assert [s['price_mode'] for s in om.submitted] == ['open', 'peer', 'peer']


def test_sell_reprice_waits_for_released_shares(monkeypatch):
    """卖单撤成终态后仍等待可用股份释放，并只重挂当前可用量。"""
    clock = _FakeClock(datetime.combine(date.today(), dt_time(9, 30)))
    monkeypatch.setattr('trading.executor.time', clock)
    monkeypatch.setattr('trading.executor.is_current_trading', lambda _: True)

    trader = _FakeTrader()
    trader.on_order = lambda code, shares, oid: SimpleNamespace(
        stock_code=code, order_type=xtconstant.STOCK_SELL,
        order_status=xtconstant.ORDER_REPORTED,
        traded_volume=0, order_volume=shares, traded_price=0.0, status_msg='')
    ex = RebalanceExecutor(trader)
    code = '600001.SH'
    om = OrderMonitor(
        ex,
        sell_orders=[(code, 500)],
        buy_allocations={},
        buy_n_stocks=[],
        prices={code: 10.0},
        signal_date=date.today(),
        trade_date=date.today(),
    )
    om.record_action = lambda *a, **k: None
    om._submit('SELL', code, 500)

    clock.current = datetime.combine(date.today(), dt_time(9, 31)).timestamp()
    assert om._cancel_expired_orders(clock.time()) is True
    om._handle_terminal_orders()
    assert om._retry_sells() is False

    trader.can_use[code] = 300
    assert om._retry_sells() is True
    assert [(shares, price) for _, shares, _, price in trader.submitted] == [
        (500, 10.0),
        (300, None),
    ]


def test_reprice_waits_for_cancel_terminal_without_duplicate_cancel(monkeypatch):
    """撤单请求已受理但仍在过渡态时，不重复撤单也不提前重挂。"""
    clock = _FakeClock(datetime.combine(date.today(), dt_time(9, 30)))
    monkeypatch.setattr('trading.executor.time', clock)
    monkeypatch.setattr('trading.executor.is_current_trading', lambda _: True)

    class _AsyncCancelTrader(_FakeTrader):
        def cancel_order(self, oid):
            self.cancelled.append(oid)
            self.orders[oid].order_status = xtconstant.ORDER_REPORTED_CANCEL

    trader = _AsyncCancelTrader(cash=1e7)
    trader.on_order = lambda code, shares, oid: SimpleNamespace(
        stock_code=code, order_type=xtconstant.STOCK_BUY,
        order_status=xtconstant.ORDER_REPORTED,
        traded_volume=0, order_volume=shares, traded_price=0.0, status_msg='')
    ex = RebalanceExecutor(trader)
    code = '600002.SH'
    om = OrderMonitor(
        ex,
        sell_orders=[],
        buy_allocations={code: 100},
        buy_n_stocks=[code],
        prices={code: 10.0},
        signal_date=date.today(),
        trade_date=date.today(),
    )
    om.record_action = lambda *a, **k: None
    om._submit('BUY', code, 100)

    clock.current = datetime.combine(date.today(), dt_time(9, 31)).timestamp()
    assert om._cancel_expired_orders(clock.time()) is True
    assert om._cancel_expired_orders(clock.time()) is False
    assert om._submit_affordable_buys() is False
    assert trader.cancelled == [1000]
    assert len(trader.submitted) == 1

    # QMT 可能在“撤单请求已受理”后异步报错并恢复为已报；超时后必须重试。
    trader.orders[1000].order_status = xtconstant.ORDER_REPORTED
    clock.current += ex.CANCEL_REQUEST_RETRY_SEC
    assert om._cancel_expired_orders(clock.time()) is True
    assert trader.cancelled == [1000, 1000]

    trader.orders[1000].order_status = xtconstant.ORDER_CANCELED
    om._handle_terminal_orders()
    assert om._submit_affordable_buys() is True
    assert trader.submitted[-1][3] is None


def test_peer_reprice_pauses_for_lunch_and_stops_before_close_auction(monkeypatch):
    monkeypatch.setattr('trading.executor.is_current_trading', lambda _: True)
    ex = RebalanceExecutor(_FakeTrader())
    code = '600000.SH'
    om = OrderMonitor(
        ex,
        sell_orders=[],
        buy_allocations={code: 100},
        buy_n_stocks=[code],
        prices={code: 10.0},
        signal_date=date.today(),
        trade_date=date.today(),
    )
    at_1101 = datetime.combine(date.today(), dt_time(11, 1)).timestamp()
    assert datetime.fromtimestamp(om._peer_reprice_at(at_1101)).time() == dt_time(13, 1)
    at_1456 = datetime.combine(date.today(), dt_time(14, 56)).timestamp()
    assert om._can_cancel_now(at_1456) is False
    at_1457 = datetime.combine(date.today(), dt_time(14, 57)).timestamp()
    assert om._can_submit_now(at_1457) is False
    assert ex.BUY_MONITOR_END == dt_time(15, 0)


def test_expired_peer_is_kept_during_close_cancel_buffer(monkeypatch):
    """14:56 后保留旧单，避免撤成后跨入 14:57 而无法再挂市价单。"""
    clock = _FakeClock(datetime.combine(date.today(), dt_time(14, 56, 59)))
    monkeypatch.setattr('trading.executor.time', clock)
    monkeypatch.setattr('trading.executor.is_current_trading', lambda _: True)
    trader = _FakeTrader(cash=1e7)
    trader.on_order = lambda code, shares, oid: SimpleNamespace(
        stock_code=code, order_type=xtconstant.STOCK_BUY,
        order_status=xtconstant.ORDER_REPORTED,
        traded_volume=0, order_volume=shares, traded_price=0.0, status_msg='')
    ex = RebalanceExecutor(trader)
    code = '600000.SH'
    om = OrderMonitor(
        ex,
        sell_orders=[],
        buy_allocations={code: 100},
        buy_n_stocks=[code],
        prices={code: 10.0},
        signal_date=date.today(),
        trade_date=date.today(),
    )
    om.record_action = lambda *a, **k: None
    om._submit('BUY', code, 100)
    om.submitted[0]['reprice_at'] = clock.time() - 1

    assert om._cancel_expired_orders(clock.time()) is False
    assert trader.cancelled == []


def test_summary_skips_replaced_intentional_reprice_cancel(monkeypatch):
    monkeypatch.setattr('trading.executor.get_stock_detail', lambda _: {})
    trader = _FakeTrader()
    trader.orders[1000] = SimpleNamespace(
        order_status=xtconstant.ORDER_CANCELED,
        traded_volume=0, traded_price=0.0, order_volume=100, status_msg='')
    trader.orders[1001] = SimpleNamespace(
        order_status=xtconstant.ORDER_SUCCEEDED,
        traded_volume=100, traded_price=10.0, order_volume=100, status_msg='')
    ex = RebalanceExecutor(trader)
    failures = []
    ex._send_failure_summary_card = lambda rows: failures.extend(rows)

    ex._summarize([
        {
            'code': '600000.SH',
            'order_id': 1000,
            'order_type': 'BUY',
            'shares': 100,
            'target_shares': 100,
            'intentional_cancel': True,
        },
        {
            'code': '600000.SH',
            'order_id': 1001,
            'order_type': 'BUY',
            'shares': 100,
            'target_shares': 100,
        },
    ])

    assert failures == []


def test_summary_reports_unreplaced_intentional_cancel(monkeypatch):
    monkeypatch.setattr('trading.executor.get_stock_detail', lambda _: {})
    trader = _FakeTrader()
    trader.orders[1000] = SimpleNamespace(
        order_status=xtconstant.ORDER_CANCELED,
        traded_volume=0, traded_price=0.0, order_volume=100, status_msg='')
    ex = RebalanceExecutor(trader)
    failures = []
    ex._send_failure_summary_card = lambda rows: failures.extend(rows)

    ex._summarize([{
        'code': '600000.SH',
        'order_id': 1000,
        'order_type': 'BUY',
        'shares': 100,
        'target_shares': 100,
        'intentional_cancel': True,
    }])

    assert len(failures) == 1


def test_summary_reports_target_gap_after_partial_replacement(monkeypatch):
    """后续替代单只覆盖一部分撤单余量时，剩余目标必须进入失败汇总。"""
    monkeypatch.setattr('trading.executor.get_stock_detail', lambda _: {})
    trader = _FakeTrader()
    trader.orders[1000] = SimpleNamespace(
        order_status=xtconstant.ORDER_CANCELED,
        traded_volume=0, traded_price=0.0, order_volume=1000, status_msg='')
    trader.orders[1001] = SimpleNamespace(
        order_status=xtconstant.ORDER_SUCCEEDED,
        traded_volume=500, traded_price=10.0, order_volume=500, status_msg='')
    ex = RebalanceExecutor(trader)
    failures = []
    ex._send_failure_summary_card = lambda rows: failures.extend(rows)

    ex._summarize([
        {
            'code': '600000.SH', 'order_id': 1000,
            'order_type': 'BUY', 'shares': 1000, 'target_shares': 1000,
            'intentional_cancel': True,
        },
        {
            'code': '600000.SH', 'order_id': 1001,
            'order_type': 'BUY', 'shares': 500, 'target_shares': 1000,
        },
    ])

    gaps = [row for row in failures if row['status'] == '目标未覆盖']
    assert len(gaps) == 1
    assert gaps[0]['shares'] == 500


def test_submit_affordable_buys_kcb_300_share_target_submits_300():
    """executor 主循环也必须保留科创板 300 股合法买入量。"""
    ex = RebalanceExecutor(_FakeTrader(cash=1e7))
    code = '688420.SH'
    om = _make_order_monitor(ex, {code: 300}, {code: 22.33}, limit_prices={code: 26.80})
    assert om._submit_affordable_buys() is True
    assert ex.trader.submitted[0][1] == 300
    assert om._remaining(code) == 0


def test_submit_affordable_buys_kcb_affordable_250_submits_250():
    """科创板资金只够 250 股时，250 股合法，不应再按 200 股截断。"""
    code = '688420.SH'
    limit_price = 26.80
    ex = RebalanceExecutor(_FakeTrader())
    unit_cost = limit_price * (1 + ex.BUY_FEE_RATE)
    ex.trader.cash = unit_cost * 250 + 0.01
    om = _make_order_monitor(ex, {code: 300}, {code: 22.33}, limit_prices={code: limit_price})
    assert om._submit_affordable_buys() is True
    assert ex.trader.submitted[0][1] == 250
    assert om._remaining(code) == 50


def test_submit_affordable_buys_kcb_remaining_below_minimum_skips():
    """科创板剩余缺口低于 200 股时不能单独提交一笔新买单。"""
    ex = RebalanceExecutor(_FakeTrader(cash=1e7))
    code = '688420.SH'
    om = _make_order_monitor(ex, {code: 199}, {code: 22.33}, limit_prices={code: 26.80})
    assert om._submit_affordable_buys() is False
    assert ex.trader.submitted == []


def test_off_hours_fast_restores_default_timeouts():
    ex = RebalanceExecutor(_FakeTrader())
    before = ex._snapshot_timeouts()
    ex._apply_off_hours_timeouts()
    assert ex.BUY_MONITOR_DEADLINE_SEC == 0.5
    assert ex.SETTLE_WAIT_SEC == 10
    ex._restore_timeouts(before)
    assert ex.BUY_MONITOR_DEADLINE_SEC is None
    assert ex.SETTLE_WAIT_SEC == 5


def test_order_reject_retry_sec_is_60():
    ex = RebalanceExecutor(_FakeTrader())
    assert ex.ORDER_REJECT_RETRY_SEC == 60.0


def test_protect_price_rounds_to_tick():
    ex = RebalanceExecutor(_FakeTrader())
    assert ex._protect_price(25.02, 0.015, 'BUY') == 25.40
    assert ex._protect_price(10.0, 0.015, 'BUY') == 10.15
    assert ex._protect_price(0.50, 0.015, 'BUY') == 0.507


def test_submit_buy_uses_protect_limit_price():
    """买入应带保护限价(开盘×1.015),不再裸对手价市价单。"""
    ex = RebalanceExecutor(_FakeTrader())
    open_px = 25.02
    code = '301520.SZ'
    expected = ex._protect_price(open_px, ex.BUY_PROTECT_PCT, 'BUY')
    om = _make_order_monitor(ex, {code: 100}, {code: open_px})
    om._submit('BUY', code, 100)
    assert len(ex.trader.submitted) == 1
    _, shares, _, limit_price = ex.trader.submitted[0]
    assert shares == 100
    assert limit_price == expected


def test_submit_buy_keeps_kcb_300_share_order():
    """科创板最低 200 股起、1 股递增；300 股买单不能被截成 200。"""
    ex = RebalanceExecutor(_FakeTrader())
    code = '688420.SH'
    om = _make_order_monitor(ex, {code: 300}, {code: 22.33})
    om._submit('BUY', code, 300)
    assert len(ex.trader.submitted) == 1
    _, shares, _, _ = ex.trader.submitted[0]
    assert shares == 300


def test_sell_reject_retries_after_backoff():
    """卖单被拒后 60s 低频重试。"""
    trader = _FakeTrader(cash=1e7)
    trader.on_order = lambda code, shares, oid: SimpleNamespace(
        order_status=xtconstant.ORDER_JUNK, traded_volume=0,
        order_volume=shares, traded_price=0.0, order_type=xtconstant.STOCK_SELL,
        status_msg='price error')
    ex = RebalanceExecutor(trader)
    om = _make_order_monitor(ex, sell_orders=[('600000.SH', 500)])
    om._submit('SELL', '600000.SH', 500)
    assert len(om.submitted) == 1
    om._handle_terminal_orders()
    assert om._sell_remaining('600000.SH') == 500
    assert om.retry_after['600000.SH'] > 0
    om.retry_after['600000.SH'] = 0
    trader.can_use['600000.SH'] = 500
    assert om._retry_sells() is True
    assert len(om.submitted) == 2
    assert [x[1] for x in trader.submitted] == [500, 500]


def test_unified_sell_buy_run():
    """完整 run() 流程：卖->买->收尾，资金够时一瞬间全发。"""
    trader = _FakeTrader(cash=1e7)
    ex = RebalanceExecutor(trader)
    ex.OPEN_LIMIT_DEADLINE_SEC = 24 * 60 * 60
    ex.BUY_MONITOR_DEADLINE_SEC = 0.1
    om = OrderMonitor(
        ex,
        sell_orders=[('600001.SH', 100), ('600002.SH', 200)],
        buy_allocations={'600003.SH': 300, '600004.SH': 400},
        buy_n_stocks=['600003.SH', '600004.SH'],
        prices={'600001.SH': 10.0, '600002.SH': 10.0, '600003.SH': 10.0, '600004.SH': 10.0},
        signal_date=date.today(), trade_date=date.today(),
        limit_prices={'600003.SH': 11.0, '600004.SH': 11.0},
    )
    om.record_action = lambda *a, **k: None
    submitted = om.run()
    assert len(submitted) == 4
    assert om._sell_remaining('600001.SH') == 0
    assert om._sell_remaining('600002.SH') == 0
    assert om._remaining('600003.SH') == 0
    assert om._remaining('600004.SH') == 0
