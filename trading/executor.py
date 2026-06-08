"""调仓下单执行器。

把 `before_trade` 算好的 `pending_rebalance` 计划落到 QMT 委托,是实盘买卖动作的唯一入口。

执行模型(09:30 尽快开火 + 事后可复盘):

1. 输入计划
   - `pending['sell_orders']`: 需卖出的 `(code, shares)`,`shares < 0` 表示清仓。
   - `pending['buy_allocations']`: 每只目标买入股数(已含 reserve)。
   - `pending['buy_n_stocks']`: 买入优先级顺序(topN)。
   - `pending['prices']`: 盘前开盘估算价。
   - `pending['limit_prices']`: 每只涨停价(前收×(1+板块涨跌幅)),买单按它估资金冻结。

2. 卖、买串行三阶段
   - Phase 1:串行提交全部卖单,不监控,让回款尽早开始到账
   - Phase 2:买入补单循环,逐只查 QMT 可用资金;卖单未全终态时延长补单窗口等回款
   - Phase 3:卖单收尾监控,超时撤余重挂。卖出回款由券商自动进「QMT 可用资金」。

3. 买单:串行、一只一只、只认 QMT 可用资金
   - 多轮扫 topN;每只:`shares = floor(min(剩余, QMT可用/涨停价)/手)×手`。
   - 钱不够买这只一手 → 跳过(留待下轮,卖单回款到了再补)。
   - 提交 1 笔市价单,等回调终态或 `BUY_ORDER_WAIT_SEC=1s`:
       · 成交        → 跳下一只(剩余下轮补,保持 topN 顺序)
       · 废单(0成交) → 当前票「减一手」重试,直到买进 / 减到买不起一手 / 连续
                        `BUY_REJECT_LIMIT=3` 次熔断该票
       · 超时(未终态)→ 撤单(剩余下轮重下),跳下一只
   - 无需本地账本:串行 + 等终态时 QMT 可用即权威值;在途单的冻结 QMT 也已扣除,跳过安全。
   - 软超时 `BUY_TIMEOUT_SEC` 后若卖单已终态且本轮零进展则收尾;硬超时 `BUY_TIMEOUT_HARD_SEC` 兜底。

4. 资金口径
   - 买入校验用「可用资金」,`_available_buy_cash()` 优先 `current_balance`/`fetch_balance`,
     再回退 `cash - frozen_cash`。
   - 单笔占用按涨停价估(券商市价单按涨停价冻结),避免按开盘价低估 → 「资金可用数不足」废单。

5. 飞书与本地留存
   - QMT 原始回调由 `watcher.py` 落 events;executor 的提交/撤单/废单/熔断落 `execution_action`。
   - 失败委托聚合成一张飞书卡(`_send_failure_summary_card`),不逐条 ERROR 刷屏。

6. 收尾:`_wait_terminal` 等终态沉淀 → `_summarize` 汇总。
"""
import time


from xtquant import xtconstant

from data.db import get_stock_detail
from trading.helper import get_order_status_label
from trading.logger import trading_logger
from trading.persistence import (
    EVT_EXECUTION_ACTION,
    SRC_EXECUTOR,
    live_trade_recorder,
)
from utils.recorder import recorder
from utils.stock.info import min_buy_shares

TERMINAL_STATUS = {
    xtconstant.ORDER_SUCCEEDED, xtconstant.ORDER_CANCELED,
    xtconstant.ORDER_JUNK, xtconstant.ORDER_PART_CANCEL,
}


# QMT 资金不足废单码:on_stock_order 回调 / query_order().status_msg 里只给数字码
# (形如 "-150906130|1|-150906130"),真正的中文「资金可用数不足」只在 on_order_error
# 的 error_msg 里。执行器靠 query_order 拿不到中文,故必须同时识别这个数字码,
# 否则资金不足废单会被误判成硬废单 → 计入熔断 → 卖单回款到账前就放弃该票。
QMT_INSUFFICIENT_FUNDS_CODE = '150906130'


def _is_insufficient_funds(msg: str) -> bool:
    """废单原因是否为「资金可用数不足」类——这类只需减仓重试,不算坏票。"""
    if not msg:
        return False
    m = str(msg)
    return (QMT_INSUFFICIENT_FUNDS_CODE in m or '资金' in m
            or '不足' in m or 'insufficient' in m.lower())


class BaseMonitor:
    def __init__(self, executor, signal_date, trade_date):
        self.executor = executor
        self.trader = executor.trader
        self.signal_date = signal_date
        self.trade_date = trade_date
        self.submitted: list[dict] = []

    def record_action(self, action: str, *, code: str = '', order_id: int = 0,
                      order_type: int | None = None, shares: int = 0,
                      amount: float = 0.0, msg: str = ''):
        try:
            live_trade_recorder.record_event(
                EVT_EXECUTION_ACTION,
                source=SRC_EXECUTOR,
                trade_date=self.trade_date,
                code=code,
                order_id=order_id,
                order_type=order_type,
                order_volume=shares,
                amount=amount,
                status_msg=f"{action}: {msg}".strip(),
            )
        except Exception as e:
            trading_logger.warning(f"execution_action 落盘失败: {e}")

    @staticmethod
    def _remaining_from_order(o):
        if not o:
            return 0
        return max(0, int(getattr(o, 'order_volume', 0) or 0) - int(getattr(o, 'traded_volume', 0) or 0))


class SellMonitor(BaseMonitor):
    """卖单 monitor:先并发卖,再监督未完成卖单,必要时撤余重挂。

    卖出不冻结现金、只释放,故并发提交无副作用;回款由券商自动进 QMT 可用资金,
    买单 monitor 每步重查 QMT 可用即可拿到,无需本地累加。
    """

    def __init__(self, executor, sell_orders, signal_date, trade_date):
        super().__init__(executor, signal_date, trade_date)
        self.sell_orders = list(sell_orders or [])
        self.retry_counts: dict[str, int] = {}
        # 供 BuyMonitor 判断卖单回款是否仍在途(决定是否延长补单窗口)
        self.finished = False

    def run(self):
        if not self.sell_orders:
            self.finished = True
            return []
        self.record_action('sell_monitor_start', msg=f"{len(self.sell_orders)} orders")
        self.submitted.extend(self._submit_initial_sells())
        self._monitor_sells()
        self.finished = True
        self.record_action('sell_monitor_done', msg=f"{len(self.submitted)} submitted")
        return self.submitted

    def _submit_initial_sells(self):
        submitted = []
        for code, shares in self.sell_orders:
            r = self.executor._submit_sell_order(code, shares,
                                                  self.signal_date, self.trade_date)
            if r:
                r['submitted_at'] = time.time()
                submitted.append(r)
                self.record_action(
                    'sell_submit',
                    code=r['code'],
                    order_id=r['order_id'],
                    order_type=xtconstant.STOCK_SELL,
                    shares=r['shares'],
                )
        return submitted

    def _monitor_sells(self):
        deadline = time.time() + self.executor.SELL_MONITOR_SEC
        while time.time() < deadline:
            active = False
            for s in list(self.submitted):
                o = self.trader.query_order(s['order_id'])
                if not o or o.order_status in TERMINAL_STATUS:
                    continue
                active = True
                age = time.time() - s.get('submitted_at', time.time())
                if age >= self.executor.SELL_ORDER_TTL_SEC:
                    self._try_repost_sell(s, o)
            if not active:
                return
            time.sleep(self.executor.MONITOR_POLL_SEC)

    def _sellable_shares(self, code, want: int) -> int:
        """按当前持仓可用股数对重挂量封顶(整百)。撤单异步,股份要等委托终态才回到
        can_use_volume;用它封顶即可保证不会重挂超过可卖量 → 不再撞「股份可用数不足」。"""
        try:
            pos = self.trader.query_stock_position(code)
        except Exception:
            pos = None
        avail = int(getattr(pos, 'can_use_volume', 0) or 0) if pos else 0
        return (min(int(want), avail) // 100) * 100

    def _try_repost_sell(self, submitted, order):
        code, oid = submitted['code'], submitted['order_id']
        if self.retry_counts.get(code, 0) >= self.executor.SELL_RETRY_LIMIT:
            return
        remaining = self._remaining_from_order(order)
        if remaining < 100:
            return
        self.retry_counts[code] = self.retry_counts.get(code, 0) + 1
        try:
            self.trader.cancel_order(oid)
            self.record_action('sell_cancel_for_repost', code=code, order_id=oid,
                               order_type=xtconstant.STOCK_SELL, shares=remaining,
                               msg=f"retry={self.retry_counts[code]}")
        except Exception as e:
            self.record_action('sell_cancel_failed', code=code, order_id=oid,
                               order_type=xtconstant.STOCK_SELL, shares=remaining, msg=str(e))
            return
        # 撤单异步:轮询等未成交股份释放回可用,再按可用量重挂(2026-06-02 09:30:14 复现)。
        deadline = time.time() + self.executor.SELL_CANCEL_SETTLE_SEC
        sellable = self._sellable_shares(code, remaining)
        while sellable < 100 and time.time() < deadline:
            time.sleep(self.executor.ORDER_POLL_SEC)
            sellable = self._sellable_shares(code, remaining)
        if sellable < 100:
            self.record_action('sell_repost_skip', code=code, order_id=oid,
                               order_type=xtconstant.STOCK_SELL, shares=remaining,
                               msg='可用股数不足,放弃重挂')
            return
        r = self.executor._submit_sell_order(code, sellable, self.signal_date, self.trade_date)
        if r:
            r['submitted_at'] = time.time()
            self.submitted.append(r)
            self.record_action('sell_repost', code=code, order_id=r['order_id'],
                               order_type=xtconstant.STOCK_SELL, shares=sellable,
                               msg=f"from={oid}")


class BuyMonitor(BaseMonitor):
    """买单 monitor:串行单循环,一只一只补到目标,只认 QMT 可用资金。

    每轮按 topN 顺序扫一遍;对每只在 `_fill_one_stock` 里:按涨停价算可买手数下单,
    废单则减一手重试到买进 / 买不起一手 / 熔断,成交或超时则跳下一只。
    """

    def __init__(self, executor, buy_allocations, buy_n_stocks, prices,
                 signal_date, trade_date, limit_prices=None):
        super().__init__(executor, signal_date, trade_date)
        self.buy_allocations = buy_allocations or {}
        self.buy_seq = [
            c for c in (buy_n_stocks or list(self.buy_allocations.keys()))
            if c in self.buy_allocations and prices.get(c, 0) > 0
        ]
        self.prices = prices or {}
        # 市价买单券商按涨停价冻结资金,用涨停价(前收×(1+板块涨跌幅))估占用,缺失回退 开盘价×SLIPPAGE
        self.limit_prices = limit_prices or {}
        self.targets = {c: int(self.buy_allocations[c]) for c in self.buy_seq}
        self.orders_by_code = {c: [] for c in self.buy_seq}
        self.fail_counts = {c: 0 for c in self.buy_seq}
        self.blocked_codes = set()
        # 并发卖买时由 execute() 注入,用于判断卖单回款是否仍在途
        self.sell_monitor: SellMonitor | None = None

    def run(self):
        if not self.buy_seq:
            return []
        self.record_action('buy_monitor_start', msg=f"{len(self.buy_seq)} targets")
        self._buy_loop()
        if self.blocked_codes:
            self.record_action('buy_blocked_codes', msg=','.join(sorted(self.blocked_codes)))
            trading_logger.warning(f"买入熔断标的: {sorted(self.blocked_codes)}")
        self.record_action('buy_monitor_done', msg=f"{len(self.submitted)} submitted")
        return self.submitted

    # ── 资金 / 缺口 ─────────────────────────────────────────
    def _available_cash(self) -> float:
        return self.executor._available_buy_cash(self.trader.query_asset())

    def _unit_cost(self, code) -> float:
        """单笔每股占用现金:优先涨停价×(1+费率)(镜像券商按涨停价冻结),缺失回退 开盘价×SLIPPAGE。"""
        lp = self.limit_prices.get(code, 0)
        if lp and lp > 0:
            return lp * (1 + self.executor.BUY_FEE_RATE)
        return self.prices[code] * self.executor.SLIPPAGE * (1 + self.executor.BUY_FEE_RATE)

    def _sells_pending(self) -> bool:
        sm = self.sell_monitor
        return bool(sm is not None and not sm.finished)

    def _filled_inflight(self, code):
        filled = inflight = 0
        for oid in self.orders_by_code[code]:
            o = self.trader.query_order(oid)
            if not o:
                continue
            traded = int(getattr(o, 'traded_volume', 0) or 0)
            filled += traded
            if o.order_status not in TERMINAL_STATUS:
                inflight += max(0, int(getattr(o, 'order_volume', 0) or 0) - traded)
        return filled, inflight

    def _remaining(self, code):
        filled, inflight = self._filled_inflight(code)
        return max(0, self.targets[code] - filled - inflight)

    def _all_done_or_blocked(self):
        for code in self.buy_seq:
            if code in self.blocked_codes:
                continue
            if self._remaining(code) >= min_buy_shares(code):
                return False
        return True

    def _can_afford_any(self, cash: float) -> bool:
        """可用现金是否买得起任意剩余标的最小1手。"""
        for code in self.buy_seq:
            if code in self.blocked_codes:
                continue
            ml = min_buy_shares(code)
            if self._remaining(code) < ml:
                continue
            if int(cash / self._unit_cost(code) / ml) * ml >= ml:
                return True
        return False

    # ── 主循环 ─────────────────────────────────────────────
    def _buy_loop(self):
        soft = time.time() + self.executor.BUY_TIMEOUT_SEC
        hard = time.time() + self.executor.BUY_TIMEOUT_HARD_SEC
        last_log = time.time()
        while time.time() < hard:
            if time.time() - last_log >= 30:
                trading_logger.info(
                    f"[BuyMonitor] 补单进行中: 已提交 {len(self.submitted)} 笔, "
                    f"卖单回款在途={self._sells_pending()}")
                last_log = time.time()
            progressed = False
            for code in self.buy_seq:
                if time.time() >= hard:
                    break
                if code in self.blocked_codes:
                    continue
                if self._remaining(code) < min_buy_shares(code):
                    continue
                if self._fill_one_stock(code):
                    progressed = True
            if self._all_done_or_blocked():
                return
            if not progressed:
                # 本轮零进展:卖单已全终态(没新回款)则收尾,否则等回款再轮
                if not self._sells_pending():
                    if time.time() >= soft:
                        return
                    # 资金不足以买入任何剩余标的的最小1手,不等了
                    cash = self._available_cash()
                    if not self._can_afford_any(cash):
                        trading_logger.warning(
                            f"[BuyMonitor] 可用资金 {cash:.2f} 不足以买入任何剩余标的,提前结束")
                        return
                time.sleep(self.executor.MONITOR_POLL_SEC)

    def _fill_one_stock(self, code) -> bool:
        """单只标的:按涨停价算可买手数下单;废单减一手重试,直到买进/买不起一手/熔断;
        成交或超时则结束本次(剩余下轮按 topN 顺序再补)。返回本次是否有成交或在途。
        """
        min_lot = min_buy_shares(code)
        cap = self._remaining(code)  # 本次尝试上限,废单后逐手下调
        while code not in self.blocked_codes:
            rem = self._remaining(code)
            if rem < min_lot:
                return False
            cap = min(cap, rem)
            cash = self._available_cash()
            afford = int(cash / self._unit_cost(code) / min_lot) * min_lot
            shares = (min(cap, afford) // min_lot) * min_lot
            if shares < min_lot:
                return False  # 现金买不起这只一手 → 交给下一只 / 下一轮
            r = self._submit_buy(code, shares)
            if not r:
                if self._register_reject(code, 'submit returned None'):
                    return False
                cap = shares - min_lot  # 减一手重试
                continue
            outcome, o = self._await_order(r['order_id'])
            if outcome == 'filled':
                return True  # 成交 → 跳下一只(剩余下轮补,保持 topN 顺序)
            if outcome == 'timeout':
                self._cancel(r['order_id'])  # 在途未终态 → 撤单,剩余下轮重下
                return True
            # 废单(0 成交)
            msg = (getattr(o, 'status_msg', '') or '').strip() if o else ''
            # 资金不足,或卖单回款仍在途(此刻钱不够只是暂时的) → 不是坏票:减一手重试,
            # 减到买不起一手就跳下一只(不计熔断)。卖单回款在途的兜底是格式无关的:即使
            # QMT 废单码变化、status_msg 拿不到中文,也能保证「等回款再补」的设计意图不被
            # 早熔断打断(今天 09:30:06 就把 4 只票熔断、回款 09:30:22 才到 → 15w 没买进)。
            if _is_insufficient_funds(msg) or self._sells_pending():
                self.record_action('buy_underfunded', code=code,
                                   order_type=xtconstant.STOCK_BUY, shares=shares, msg=msg)
                cap = shares - min_lot
                continue
            if self._register_reject(code, msg):
                return False
            cap = shares - min_lot  # 非资金类硬废单:计熔断并减一手重试
        return False

    def _submit_buy(self, code, shares):
        ml = min_buy_shares(code)
        shares = int(shares // ml * ml)
        if shares < ml:
            return None
        r = self.executor._submit_buy_order(code, shares, self.signal_date, self.trade_date)
        if not r:
            return None
        r['submitted_at'] = time.time()
        self.orders_by_code[code].append(r['order_id'])
        self.submitted.append(r)
        self.record_action(
            'buy_submit', code=code, order_id=r['order_id'],
            order_type=xtconstant.STOCK_BUY, shares=shares,
            amount=shares * self._unit_cost(code),
        )
        return r

    def _await_order(self, order_id):
        """轮询等该委托终态,最多 BUY_ORDER_WAIT_SEC。
        返回 ('filled'|'reject'|'timeout', order)。filled=终态且有成交,reject=终态零成交。
        """
        deadline = time.time() + self.executor.BUY_ORDER_WAIT_SEC
        last = None
        while time.time() < deadline:
            o = self.trader.query_order(order_id)
            last = o
            if o and o.order_status in TERMINAL_STATUS:
                traded = int(getattr(o, 'traded_volume', 0) or 0)
                return ('filled' if traded > 0 else 'reject'), o
            time.sleep(self.executor.ORDER_POLL_SEC)
        return 'timeout', last

    def _cancel(self, order_id):
        try:
            self.trader.cancel_order(order_id)
            self.record_action('buy_cancel_timeout', order_id=order_id,
                               order_type=xtconstant.STOCK_BUY, msg='inflight>wait')
        except Exception as e:
            self.record_action('buy_cancel_failed', order_id=order_id,
                               order_type=xtconstant.STOCK_BUY, msg=str(e))

    def _register_reject(self, code, msg='') -> bool:
        """记一次废单;连续达 BUY_REJECT_LIMIT 次则熔断该票。返回是否已熔断。"""
        self.fail_counts[code] += 1
        self.record_action('buy_reject', code=code,
                           order_type=xtconstant.STOCK_BUY, msg=msg)
        if self.fail_counts[code] >= self.executor.BUY_REJECT_LIMIT:
            self.blocked_codes.add(code)
            trading_logger.warning(
                f"买入熔断: {code} 连续{self.fail_counts[code]}次失败,放弃该票并继续后续。原因: {msg}")
            return True
        return False


class RebalanceExecutor:
    """调仓下单执行器,依赖一个 `Trader` 实例进行实际委托。"""

    SLIPPAGE = 1.01       # 市价单实际成交价高于估算开盘价的缓冲(仅在缺涨停价时回退用)
    BUY_FEE_RATE = 0.0000854 + 0.00002  # 买入手续费率(佣金+过户费),与回测 fee 口径对齐
    BUY_TIMEOUT_SEC = 120       # 买入补单软超时:卖单已全终态且本轮零进展则收尾
    BUY_TIMEOUT_HARD_SEC = 600  # 硬超时:即便卖单回款仍未到账也必须收尾,防死循环
    BUY_ORDER_WAIT_SEC = 1.0    # 单笔买单等回调终态的最长秒数,超时则撤单跳下一只
    BUY_REJECT_LIMIT = 3        # 同一标的连续废单达此次数后熔断,放弃该票
    SETTLE_WAIT_SEC = 30        # 等待所有委托进入终态的最长秒数
    SELL_MONITOR_SEC = 120
    SELL_ORDER_TTL_SEC = 12
    SELL_RETRY_LIMIT = 1
    SELL_CANCEL_SETTLE_SEC = 3.0  # 重挂前等原卖单撤单进入终态(股份释放回可用)的最长秒数
    MONITOR_POLL_SEC = 0.3      # 买/卖主循环每轮间隔
    ORDER_POLL_SEC = 0.1        # 单笔买单等回调的轮询间隔
    # --skip 且真实时钟在闭市: order_stock 同步 API 可阻塞数分钟(非「废单慢」);
    # 此时跳过真实委托,仅收口战报/盘后。盘中委托仍走正常轮询超时。
    _TIMEOUT_FIELDS = (
        'BUY_TIMEOUT_SEC', 'BUY_TIMEOUT_HARD_SEC', 'BUY_ORDER_WAIT_SEC',
        'SETTLE_WAIT_SEC', 'SELL_MONITOR_SEC',
    )
    _OFF_HOURS_TIMEOUTS = {
        'BUY_TIMEOUT_SEC': 30,
        'BUY_TIMEOUT_HARD_SEC': 90,
        'BUY_ORDER_WAIT_SEC': 0.5,
        'SETTLE_WAIT_SEC': 10,
        'SELL_MONITOR_SEC': 30,
    }

    def __init__(self, trader):
        self.trader = trader

    def _snapshot_timeouts(self) -> dict:
        return {k: getattr(self, k) for k in self._TIMEOUT_FIELDS}

    def _restore_timeouts(self, snap: dict):
        for k, v in snap.items():
            setattr(self, k, v)

    def _apply_off_hours_timeouts(self):
        for k, v in self._OFF_HOURS_TIMEOUTS.items():
            setattr(self, k, v)

    @staticmethod
    def _available_buy_cash(asset) -> float:
        """QMT 买入校验使用可用资金,不是资金余额 cash。

        `cash` 在部分券商环境里会包含不可立即买入的资金;`current_balance` /
        `fetch_balance` 更接近柜台报错里的 available,能避免误判可买后被柜台拒单。
        """
        if not asset:
            return 0.0
        for field in ('current_balance', 'fetch_balance'):
            value = getattr(asset, field, None)
            if value is not None:
                return max(0.0, float(value))
        cash = float(getattr(asset, 'cash', 0.0) or 0.0)
        frozen = float(getattr(asset, 'frozen_cash', 0.0) or 0.0)
        return max(0.0, cash - frozen)

    def _submit_sell_order(self, code, shares, signal_date, trade_date):
        """提交卖出委托。shares=-1 表示全部清仓。返回 {code, order_type, order_id, shares} 或 None。"""
        remark = f'rebalance signal={signal_date.isoformat()} trade={trade_date.isoformat()}'
        try:
            if shares < 0:
                order_id = self.trader.clear_position(code, reason=remark)
            else:
                order_id = self.trader.order(
                    xtconstant.STOCK_SELL, code, shares, None, order_remark=remark)
            if order_id is None:
                trading_logger.info(f"{code} 无需卖出或委托未发出")
                return None
            trading_logger.info(
                f"已提交卖出委托: {code} {'全仓' if shares < 0 else f'{shares}股'} order_id={order_id}")
            recorder.mark("提交卖出委托")
            return {'code': code, 'order_type': 'SELL', 'order_id': order_id, 'shares': shares}
        except ValueError as e:
            trading_logger.info(f"{code} 卖出前校验拦截: {e}")
        except Exception as e:
            trading_logger.warning(f"{code} 卖出委托失败: {e}")
        return None

    def _submit_buy_order(self, code, shares, signal_date, trade_date):
        """提交买入委托。返回 {code, order_type, order_id, shares} 或 None。"""
        remark = f'rebalance signal={signal_date.isoformat()} trade={trade_date.isoformat()}'
        try:
            order_id = self.trader.order(
                xtconstant.STOCK_BUY, code, shares, None, order_remark=remark)
            trading_logger.info(f"已提交买入委托: {code} * {shares} 股 order_id={order_id}")
            recorder.mark("提交买入委托")
            return {'code': code, 'order_type': 'BUY', 'order_id': order_id, 'shares': shares}
        except ValueError as e:
            trading_logger.info(f"{code} 买入前校验拦截: {e}")
        except Exception as e:
            trading_logger.warning(f"{code} 买入委托失败: {e}")
        return None

    def _wait_terminal(self, submitted):
        """轮询等待所有委托进入终态。"""
        waited = 0
        pending_orders = []
        while waited < self.SETTLE_WAIT_SEC:
            pending_orders = [
                s['code'] for s in submitted
                if (o := self.trader.query_order(s['order_id'])) and o.order_status not in TERMINAL_STATUS
            ]
            if not pending_orders:
                return
            time.sleep(1)
            waited += 1
        trading_logger.warning(f"等待成交超时, 仍有{len(pending_orders)}笔未终态: {pending_orders[:5]}...")

    def _summarize(self, submitted):
        """汇总成交 / 未完成 / 失败并打印。"""
        ok_list, fail_rows, partial_list = [], [], []
        for s in submitted:
            o = self.trader.query_order(s['order_id'])
            order_status = o.order_status if o else None
            status = get_order_status_label(order_status) if o else '查询失败'
            msg = o.status_msg if o else ''
            traded = o.traded_volume if o else 0
            price = o.traded_price if o else 0
            vol = o.order_volume if o else 0
            detail = get_stock_detail(s['code'])
            name = (detail.get('InstrumentName', '') if detail else '').strip()
            label = f"{s['code']} {name}".strip()
            line = f"{s['order_type']:4s} {label} {s['shares']}股 → {status}"
            if traded and traded != vol:
                line += f" {traded}/{vol}股"
            if price:
                line += f" @{price:.2f}"
            if order_status == xtconstant.ORDER_SUCCEEDED:
                ok_list.append(line)
            elif order_status in (
                xtconstant.ORDER_JUNK,
                xtconstant.ORDER_CANCELED,
                xtconstant.ORDER_PART_CANCEL,
            ):
                fail_rows.append({
                    'order_type': s['order_type'],
                    'code': s['code'],
                    'name': name,
                    'shares': int(s['shares']),
                    'status': status,
                    'msg': msg,
                    'line': f"{line} {msg}",
                })
            else:
                partial_list.append(f"{line} {msg}")

        if ok_list:
            trading_logger.info(f"=== 成交 {len(ok_list)} 笔 ===")
            for l in ok_list:
                trading_logger.info(f"  {l}")
        if partial_list:
            trading_logger.warning(f"=== 未完成 {len(partial_list)} 笔 ===")
            for l in partial_list:
                trading_logger.warning(f"  {l}")
        if fail_rows:
            self._send_failure_summary_card(fail_rows)
            trading_logger.warning(f"=== 失败 {len(fail_rows)} 笔 ===")
            for r in fail_rows:
                trading_logger.warning(f"  {r['line']}")

    def _send_failure_summary_card(self, fail_rows):
        """失败委托只发一张飞书聚合卡,避免每笔失败触发 ERROR sink 刷屏。"""
        grouped = {}
        for r in fail_rows:
            key = (r['order_type'], r['code'], r['name'], r['status'], r['msg'])
            item = grouped.setdefault(key, {
                'direction': r['order_type'],
                'stock': f"{r['code']} {r['name']}".strip(),
                'status': r['status'],
                'count': 0,
                'shares': 0,
                'reason': r['msg'] or '-',
            })
            item['count'] += 1
            item['shares'] += r['shares']

        rows = sorted(grouped.values(), key=lambda x: (-x['count'], x['stock']))
        try:
            from trading.lark.sender import LarkMsgLevel, lark_sender
            lark_sender.send_table_card(
                title=f"❌ 调仓失败汇总 · {len(fail_rows)} 笔 / {len(rows)} 类",
                level=LarkMsgLevel.Danger,
                summary_md="失败委托已聚合展示；完整本地明细见交易日志。",
                tables=[{
                    'title': '**失败委托**',
                    'element_id': 'rebalance_failures',
                    'page_size': 20,
                    'columns': [
                        {'name': 'direction', 'display_name': '方向', 'horizontal_align': 'left'},
                        {'name': 'stock', 'display_name': '股票', 'horizontal_align': 'left'},
                        {'name': 'status', 'display_name': '状态', 'horizontal_align': 'left'},
                        {'name': 'count', 'display_name': '笔数', 'horizontal_align': 'right'},
                        {'name': 'shares', 'display_name': '股数', 'horizontal_align': 'right'},
                        {'name': 'reason', 'display_name': '原因', 'horizontal_align': 'left'},
                    ],
                    'rows': rows,
                }],
            )
        except Exception as e:
            trading_logger.warning(f"失败委托聚合卡片发送失败: {e}")

    def execute(self, pending, *, execute_sell=True, execute_buy=True,
                off_hours_fast: bool = False):
        """执行一次调仓:提交卖买委托、等待终态、汇总。

        Args:
            pending: before_trade 产出的 pending_rebalance 字典。
            execute_sell / execute_buy: 是否执行卖 / 买（分时段调仓时使用）。
            off_hours_fast: 真实时钟在闭市且 --skip 演练时缩短轮询超时,避免卡 10min+。
        """
        timeout_snap = self._snapshot_timeouts()
        self._skip_real_orders = off_hours_fast
        if off_hours_fast:
            trading_logger.warning(
                "[执行] 闭市 --skip: QMT order_stock 同步调用会长时间阻塞(实测>3min), "
                "非「秒废单」; 跳过真实买卖, 仅战报定稿+盘后 diff")
        try:
            return self._execute_impl(
                pending, execute_sell=execute_sell, execute_buy=execute_buy)
        finally:
            self._skip_real_orders = False
            self._restore_timeouts(timeout_snap)

    def _execute_impl(self, pending, *, execute_sell=True, execute_buy=True):
        signal_date = pending['signal_date']
        trade_date = pending['trade_date']
        sell_orders = pending.get('sell_orders', [])
        buy_allocations = pending.get('buy_allocations', {})
        buy_n_stocks = pending.get('buy_n_stocks', list(buy_allocations.keys()))
        prices = pending.get('prices', {})
        limit_prices = pending.get('limit_prices', {})

        trading_logger.info(
            f"开始调仓: sell={len(sell_orders)} buy={len(buy_allocations)} "
            f"signal={signal_date.isoformat()} trade={trade_date.isoformat()}")

        if getattr(self, '_skip_real_orders', False):
            self._summarize([])
            return []

        sell_submitted, buy_submitted = [], []
        try:
            avail = self._available_buy_cash(self.trader.query_asset())
            live_trade_recorder.record_event(
                EVT_EXECUTION_ACTION,
                source=SRC_EXECUTOR,
                trade_date=trade_date,
                status_msg=f"rebalance_start: available_cash={avail:.2f}",
            )
        except Exception as e:
            trading_logger.warning(f"[Executor] rebalance_start 事件记录失败: {e}")

        sell_monitor = None
        if execute_sell and sell_orders:
            sell_monitor = SellMonitor(self, sell_orders, signal_date, trade_date)
        buy_monitor = None
        if execute_buy and buy_allocations:
            buy_monitor = BuyMonitor(
                self, buy_allocations, buy_n_stocks, prices,
                signal_date, trade_date, limit_prices=limit_prices)

        # 三阶段串行：提交卖单 → 买入（边买边等回款）→ 卖单收尾
        # Phase 1: 串行提交全部卖单（不监控），让回款尽早开始到账
        if sell_monitor:
            sell_submitted = sell_monitor._submit_initial_sells()
        # Phase 2: 买入（逐只查 QMT 可用资金，sell_monitor 未终态时延长补单窗口等回款）
        if buy_monitor:
            if sell_monitor:
                buy_monitor.sell_monitor = sell_monitor
            buy_submitted = buy_monitor.run()
        # Phase 3: 卖单收尾监控（撤余重挂未成交的）
        if sell_monitor:
            sell_monitor._monitor_sells()
            sell_monitor.finished = True

        submitted = sell_submitted + buy_submitted
        self._wait_terminal(submitted)
        self._summarize(submitted)
        return submitted
