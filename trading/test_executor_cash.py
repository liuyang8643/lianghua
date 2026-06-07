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


def test_is_insufficient_funds_recognizes_qmt_numeric_code():
    """回归 2026-06-02:query_order().status_msg 只给数字码 -150906130|1|-150906130,
    不含中文「资金可用数不足」。必须靠数字码识别,否则资金废单被误判为硬废单。"""
    from trading.executor import _is_insufficient_funds
    assert _is_insufficient_funds('-150906130|1|-150906130') is True   # QMT 买入资金不足码
    assert _is_insufficient_funds('资金可用数不足,尚需3440') is True
    assert _is_insufficient_funds('insufficient buying power') is True
    assert _is_insufficient_funds('无对手方价格,废单') is False         # 真·硬废单
    assert _is_insufficient_funds('') is False


def test_fill_one_stock_numeric_code_reject_does_not_block():
    """资金不足废单只给数字码(无中文)时,也不能计熔断。"""
    trader = _FakeTrader(cash=1e7)
    trader.on_order = lambda code, shares, oid: SimpleNamespace(
        order_status=xtconstant.ORDER_JUNK, traded_volume=0,
        order_volume=shares, traded_price=0.0, status_msg='-150906130|1|-150906130')
    ex = RebalanceExecutor(trader)
    code = '688466.SH'  # 科创 min_lot=200
    bm = _make_buy_monitor(ex, {code: 600}, {code: 10.0}, limit_prices={code: 11.0})
    bm._fill_one_stock(code)
    assert code not in bm.blocked_codes
    assert bm.fail_counts[code] == 0


def test_fill_one_stock_no_block_while_sells_pending():
    """卖单回款在途时,买单废单(哪怕拿不到任何 msg)都不计熔断——格式无关的兜底。

    复刻 2026-06-02:开盘现金不够、4 只买单被废,但卖单回款 09:30:22 才到。
    旧逻辑在 09:30:06 就把它们熔断;新逻辑应保持不熔断,等回款再补。
    """
    trader = _FakeTrader(cash=1e7)
    # 模拟拿不到 status_msg 的硬废单(空 msg)——若无 sells-pending 兜底就会熔断
    trader.on_order = lambda code, shares, oid: SimpleNamespace(
        order_status=xtconstant.ORDER_JUNK, traded_volume=0,
        order_volume=shares, traded_price=0.0, status_msg='')
    ex = RebalanceExecutor(trader)
    code = '300876.SZ'  # 创业 min_lot=200
    bm = _make_buy_monitor(ex, {code: 600}, {code: 10.0}, limit_prices={code: 11.0})
    # 注入一个「未完成」的卖单 monitor,表示回款仍在途
    sm = SellMonitor(ex, [('600000.SH', 100)], date.today(), date.today())
    bm.sell_monitor = sm
    assert bm._sells_pending() is True
    bm._fill_one_stock(code)
    assert code not in bm.blocked_codes   # 回款在途不熔断
    assert bm.fail_counts[code] == 0

    # 卖单收尾后,同样的空 msg 硬废单恢复计熔断
    sm.finished = True
    bm2 = _make_buy_monitor(ex, {code: 600}, {code: 10.0}, limit_prices={code: 11.0})
    bm2.sell_monitor = sm
    bm2._fill_one_stock(code)
    assert code in bm2.blocked_codes
    assert bm2.fail_counts[code] == ex.BUY_REJECT_LIMIT


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


def _stuck_sell_monitor(ex, code, volume):
    sm = SellMonitor(ex, [(code, volume)], date.today(), date.today())
    sm.record_action = lambda *a, **k: None
    ex.trader.orders[5000] = SimpleNamespace(
        stock_code=code, order_status=xtconstant.ORDER_REPORTED,
        order_volume=volume, traded_volume=0, status_msg='')
    ex.trader.can_use[code] = 0  # 在途卖单股份被冻结
    sub = {'code': code, 'order_id': 5000, 'shares': volume, 'submitted_at': 0}
    sm.submitted.append(sub)
    return sm, sub


def test_sell_repost_waits_for_release_then_reposts_available():
    """复刻 2026-06-02:撤单后股份释放,按当前可用量重挂(不再撞「股份可用数不足」)。"""
    ex = RebalanceExecutor(_FakeSellTrader(release_on_cancel=True))
    ex.SELL_CANCEL_SETTLE_SEC = 0.5
    code = '603168.SH'
    sm, sub = _stuck_sell_monitor(ex, code, 5900)
    sm._try_repost_sell(sub, ex.trader.orders[5000])
    assert len(ex.trader.sell_submits) == 1
    rcode, rshares, _ = ex.trader.sell_submits[0]
    assert rcode == code and rshares == 5900


def test_sell_repost_skips_when_shares_not_released():
    """撤单后股份始终没释放回可用 → 放弃重挂,不盲目用旧量下单产生废单。"""
    ex = RebalanceExecutor(_FakeSellTrader(release_on_cancel=False))
    ex.SELL_CANCEL_SETTLE_SEC = 0.3
    code = '688026.SH'
    sm, sub = _stuck_sell_monitor(ex, code, 2400)
    sm._try_repost_sell(sub, ex.trader.orders[5000])
    assert ex.trader.sell_submits == []


def test_fill_one_stock_skips_when_cash_below_one_lot():
    """现金不足以按涨停价买一手 → 不下单,直接跳过(不产生废单)。"""
    # 现金只有 1500,涨停价 11 → 一手(100股)需 ~1100,但科创最小 200 手 → 需 ~2200 > 1500
    ex = RebalanceExecutor(_FakeTrader(cash=1500.0))
    code = '688296.SH'  # 科创,min_lot=200
    bm = _make_buy_monitor(ex, {code: 2000}, {code: 10.0}, limit_prices={code: 11.0})
    assert bm._fill_one_stock(code) is False
    assert len(bm.submitted) == 0  # 没下单,没废单


def test_off_hours_fast_restores_default_timeouts():
    ex = RebalanceExecutor(_FakeTrader())
    before = ex._snapshot_timeouts()
    ex._apply_off_hours_timeouts()
    assert ex.BUY_TIMEOUT_HARD_SEC == 90
    assert ex.SELL_MONITOR_SEC == 30
    ex._restore_timeouts(before)
    assert ex.BUY_TIMEOUT_HARD_SEC == 600
    assert ex.SELL_MONITOR_SEC == 120
