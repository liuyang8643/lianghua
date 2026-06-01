from datetime import date
from types import SimpleNamespace

from xtquant import xtconstant

from trading.executor import BuyMonitor, RebalanceExecutor, SellMonitor


class _FakeTrader:
    """可控 trader:order 默认全成,可用 on_order 钩子改写订单结局;query_asset 返回可用现金。"""

    def __init__(self, cash: float = 1e9):
        self.orders: dict[int, SimpleNamespace] = {}
        self.cash = cash
        self._next = 1000
        self.on_order = None  # (code, shares, oid) -> SimpleNamespace | None
        self.submitted = []

    def query_asset(self):
        return SimpleNamespace(current_balance=self.cash)

    def order(self, order_type, code, shares, price, order_remark=''):
        oid = self._next
        self._next += 1
        self.submitted.append((code, shares, oid))
        o = SimpleNamespace(order_status=xtconstant.ORDER_SUCCEEDED,
                            traded_volume=shares, order_volume=shares,
                            traded_price=0.0, status_msg='')
        if self.on_order:
            o = self.on_order(code, shares, oid) or o
        self.orders[oid] = o
        return oid

    def query_order(self, oid):
        return self.orders.get(oid)

    def cancel_order(self, oid):
        o = self.orders.get(oid)
        if o:
            o.order_status = xtconstant.ORDER_CANCELED


def _make_buy_monitor(executor, allocations, prices, limit_prices=None):
    bm = BuyMonitor(
        executor, allocations, list(allocations.keys()), prices,
        date.today(), date.today(), limit_prices=limit_prices,
    )
    bm.record_action = lambda *a, **k: None  # 隔离落盘副作用
    return bm


# ── 可用资金口径 ─────────────────────────────────────────

def test_available_buy_cash_prefers_current_balance():
    asset = SimpleNamespace(cash=100_000.0, current_balance=12_345.0,
                            fetch_balance=12_000.0, frozen_cash=5_000.0)
    assert RebalanceExecutor._available_buy_cash(asset) == 12_345.0


def test_available_buy_cash_falls_back_to_fetch_balance():
    asset = SimpleNamespace(cash=100_000.0, current_balance=None,
                            fetch_balance=23_456.0, frozen_cash=5_000.0)
    assert RebalanceExecutor._available_buy_cash(asset) == 23_456.0


def test_available_buy_cash_falls_back_to_cash_minus_frozen():
    asset = SimpleNamespace(cash=100_000.0, frozen_cash=7_000.0)
    assert RebalanceExecutor._available_buy_cash(asset) == 93_000.0


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
    bm = _make_buy_monitor(ex, {code: 200}, {code: 14.3}, limit_prices={code: 17.23})
    expected = 17.23 * (1 + ex.BUY_FEE_RATE)
    assert abs(bm._unit_cost(code) - expected) < 1e-9
    # 涨停价口径显著高于开盘价口径,避免低估资金占用
    assert bm._unit_cost(code) > 14.3 * ex.SLIPPAGE * (1 + ex.BUY_FEE_RATE)


def test_unit_cost_falls_back_to_open_when_no_limit():
    ex = RebalanceExecutor(_FakeTrader())
    code = '600000.SH'
    bm = _make_buy_monitor(ex, {code: 100}, {code: 10.0})
    expected = 10.0 * ex.SLIPPAGE * (1 + ex.BUY_FEE_RATE)
    assert abs(bm._unit_cost(code) - expected) < 1e-9


# ── 卖单在途感知 / 缺口计算 ────────────────────────────────

def test_sells_pending_reflects_sell_monitor_state():
    ex = RebalanceExecutor(_FakeTrader())
    bm = _make_buy_monitor(ex, {'600000.SH': 100}, {'600000.SH': 10.0})
    assert bm._sells_pending() is False
    sm = SellMonitor(ex, [], date.today(), date.today())
    bm.sell_monitor = sm
    assert bm._sells_pending() is True
    sm.finished = True
    assert bm._sells_pending() is False


def test_empty_sell_monitor_marks_finished():
    ex = RebalanceExecutor(_FakeTrader())
    sm = SellMonitor(ex, [], date.today(), date.today())
    assert sm.run() == []
    assert sm.finished is True


def test_remaining_counts_filled_and_inflight():
    ex = RebalanceExecutor(_FakeTrader())
    code = '600000.SH'
    bm = _make_buy_monitor(ex, {code: 1000}, {code: 10.0})
    bm.trader.orders[1] = SimpleNamespace(
        order_status=xtconstant.ORDER_SUCCEEDED, traded_volume=300, order_volume=300)
    bm.trader.orders[2] = SimpleNamespace(
        order_status=xtconstant.ORDER_REPORTED, traded_volume=0, order_volume=200)
    bm.orders_by_code[code] = [1, 2]
    assert bm._remaining(code) == 1000 - 300 - 200


# ── 串行补单核心 ─────────────────────────────────────────

def test_fill_one_stock_happy_path_fully_buys():
    """正常成交:一笔打满目标,剩余归零。"""
    ex = RebalanceExecutor(_FakeTrader(cash=1e7))
    code = '600000.SH'
    bm = _make_buy_monitor(ex, {code: 1000}, {code: 10.0}, limit_prices={code: 11.0})
    assert bm._fill_one_stock(code) is True
    assert bm._remaining(code) == 0
    assert len(bm.submitted) == 1


def test_fill_one_stock_blocks_after_hard_reject_limit():
    """非资金类硬废单(如价格/权限):减一手重试,达 BUY_REJECT_LIMIT 次后熔断该票。"""
    trader = _FakeTrader(cash=1e7)
    trader.on_order = lambda code, shares, oid: SimpleNamespace(
        order_status=xtconstant.ORDER_JUNK, traded_volume=0,
        order_volume=shares, traded_price=0.0, status_msg='无对手方价格,废单')
    ex = RebalanceExecutor(trader)
    code = '600000.SH'
    bm = _make_buy_monitor(ex, {code: 1000}, {code: 10.0}, limit_prices={code: 11.0})
    bm._fill_one_stock(code)
    assert code in bm.blocked_codes
    assert bm.fail_counts[code] == ex.BUY_REJECT_LIMIT
    assert len(bm.submitted) == 3  # 1000 → 900 → 800


def test_fill_one_stock_underfunded_reject_does_not_block():
    """资金不足废单:只减仓重试、不计熔断;减到买不起一手就跳过,不会被熔断。"""
    trader = _FakeTrader(cash=1e7)
    trader.on_order = lambda code, shares, oid: SimpleNamespace(
        order_status=xtconstant.ORDER_JUNK, traded_volume=0,
        order_volume=shares, traded_price=0.0, status_msg='资金可用数不足,尚需3440')
    ex = RebalanceExecutor(trader)
    code = '688296.SH'  # 科创 min_lot=200
    bm = _make_buy_monitor(ex, {code: 600}, {code: 10.0}, limit_prices={code: 11.0})
    bm._fill_one_stock(code)
    assert code not in bm.blocked_codes      # 资金类不熔断
    assert bm.fail_counts[code] == 0         # 不计坏票
    assert len(bm.submitted) > 0             # 确实减仓试过


def test_fill_one_stock_skips_when_cash_below_one_lot():
    """现金不足以按涨停价买一手 → 不下单,直接跳过(不产生废单)。"""
    # 现金只有 1500,涨停价 11 → 一手(100股)需 ~1100,但科创最小 200 手 → 需 ~2200 > 1500
    ex = RebalanceExecutor(_FakeTrader(cash=1500.0))
    code = '688296.SH'  # 科创,min_lot=200
    bm = _make_buy_monitor(ex, {code: 2000}, {code: 10.0}, limit_prices={code: 11.0})
    assert bm._fill_one_stock(code) is False
    assert len(bm.submitted) == 0  # 没下单,没废单
