"""调仓下单执行器。

把 `before_trade` 算好的 `pending_rebalance` 计划落到 QMT 委托,是实盘买卖动作的唯一入口。

执行模型(因子算完立即挂单 + 事后可复盘):

1. 输入计划
   - `pending['sell_orders']`: 需卖出的 `(code, shares)`,`shares < 0` 表示清仓。
   - `pending['buy_allocations']`: 每只目标买入股数(已含 reserve)。
   - `pending['buy_n_stocks']`: 买入优先级顺序(topN)。
   - `pending['prices']`: 盘前开盘估算价。
   - `pending['limit_prices']`: 每只涨停价(前收×(1+板块涨跌幅)),买单按它估资金冻结。

2. 卖、买挂单模型（统一 OrderMonitor）
   - 卖单:一次性提交全部 open[T] 限价单，被拒后低频重试。
   - 买单:按 TopN 顺序检查 QMT 可用资金,够一只/一部分就立即提交 open[T] 限价单。
   - 资金暂时不够:不下废单,轮询 QMT 可用资金;任何卖出成交回款到账后继续挂后续买单。

3. 开盘限价 + 对手价轮换
   - 首单在 09:31 前按 open[T] 限价提交；09:31 未成交则撤余并改挂对手价。
   - 后续对手价单每 30 分钟撤余重挂；午休暂停，14:56 后保留末单，15:00 停止。
   - 资金不足废单只等待回款后重试,不熔断。
   - 非资金类废单(价格范围/柜台限制)不减手、不熔断,按当前价格模式低频重试。
   - 无需本地账本:每次提交前重新查询 QMT 可用资金;在途冻结由 QMT cash 反映。

4. 资金口径
   - 买入校验用「可用资金」,`_available_buy_cash()` 使用 XtAsset.cash（可用金额,实时含冻结扣减与卖出回款）。
   - 单笔占用按涨停价估(券商市价单按涨停价冻结),避免按开盘价低估 → 「资金可用数不足」废单。

5. 飞书与本地留存
   - QMT 原始回调由 `watcher.py` 落 events;executor 的提交/资金等待/废单重试落 `execution_action`。
   - 失败委托聚合成一张飞书卡(`_send_failure_summary_card`),不逐条 ERROR 刷屏。

6. 收尾:15:00 前持续监控;未终态订单交由交易所闭市处理,回调由 watcher 记录。
"""
import time
from datetime import datetime, time as dt_time


from xtquant import xtconstant

from core.fees import LIVE_BUY_FEE_RATE
from data.db import get_stock_detail
from trading.helper import get_order_status_label
from trading.logger import trading_logger
from trading.persistence import (
    EVT_EXECUTION_ACTION,
    SRC_EXECUTOR,
    live_trade_recorder,
)
from utils.recorder import recorder
from utils.stock.info import floor_buy_shares, min_buy_shares
from utils.stock.time import is_current_trading

TERMINAL_STATUS = {
    xtconstant.ORDER_SUCCEEDED, xtconstant.ORDER_CANCELED,
    xtconstant.ORDER_JUNK, xtconstant.ORDER_PART_CANCEL,
}
CANCELABLE_STATUS = {
    xtconstant.ORDER_REPORTED, xtconstant.ORDER_PART_SUCC,
}


# QMT 资金不足废单码:on_stock_order 回调 / query_order().status_msg 里只给数字码
# (形如 "-150906130|1|-150906130"),真正的中文「资金可用数不足」只在 on_order_error
# 的 error_msg 里。执行器靠 query_order 拿不到中文,故必须同时识别这个数字码,
# 否则资金不足废单会被误判成普通柜台废单,导致等待回款语义失真。
QMT_INSUFFICIENT_FUNDS_CODE = '150906130'


def _is_insufficient_funds(msg: str) -> bool:
    """废单原因是否为「资金可用数不足」类——这类只需减仓重试,不算坏票。"""
    if not msg:
        return False
    m = str(msg)
    return (QMT_INSUFFICIENT_FUNDS_CODE in m or '资金' in m
            or '不足' in m or 'insufficient' in m.lower())


class OrderMonitor:
    """统一买卖订单监控器。

    卖单：一次性全提交，被拒后低频重试。
    买单：按 TopN + QMT 可用资金循环提交，资金不够等卖单回款。
    买卖共用：_submit / _handle_terminal_orders / retry_after 退避。
    """

    def __init__(self, executor, sell_orders, buy_allocations, buy_n_stocks,
                 prices, signal_date, trade_date, limit_prices=None):
        self.executor = executor
        self.trader = executor.trader
        self.signal_date = signal_date
        self.trade_date = trade_date
        self.submitted: list[dict] = []

        # 解析卖单：负数 → 查 QMT 取实际可卖股数
        self.sell_targets: dict[str, int] = {}
        for code, shares in (sell_orders or []):
            if shares < 0:
                pos = self.trader.query_stock_position(code)
                resolved = int(getattr(pos, 'can_use_volume', 0) or 0) if pos else 0
                self.sell_targets[code] = resolved
            else:
                self.sell_targets[code] = shares

        # 解析买单
        self.buy_allocations = buy_allocations or {}
        self.prices = prices or {}
        self.limit_prices = limit_prices or {}
        self.buy_seq = [
            c for c in (buy_n_stocks or list(self.buy_allocations))
            if c in self.buy_allocations and self.prices.get(c, 0) > 0
        ]
        self.targets = {c: int(self.buy_allocations[c]) for c in self.buy_seq}

        # 统一状态追踪（买+卖）
        self.orders_by_code: dict[str, list[int]] = {}
        self.handled_terminal: set[int] = set()
        self.retry_after: dict[str, float] = {}
        self._local_inflight: dict[int, int] = {}
        self._last_order_state: dict[int, tuple[int, int, int, int, str]] = {}
        self._submitted_by_id: dict[int, dict] = {}
        self.cancel_requested: dict[int, float] = {}
        self.reprice_pending: set[str] = set()
        override = getattr(self.executor, 'OPEN_LIMIT_DEADLINE_SEC', None)
        self.open_limit_deadline = (
            time.time() + float(override) if override is not None
            else datetime.combine(self.trade_date, self.executor.OPEN_LIMIT_END).timestamp()
        )

    # ── 主流程 ─────────────────────────────────────────────

    def run(self):
        # Phase 1: 卖单一次性全提交（已有在途则跳过，等废单退避后重试）
        if self._can_submit_now(time.time()):
            for code in self.sell_targets:
                if self._has_open_order(code):
                    continue
                rem = self._sell_remaining(code)
                if rem <= 0:
                    continue
                self._submit('SELL', code, rem)

        # Phase 2+3: 主循环（买提交 + 买卖废单重试）
        if self.buy_seq or self.sell_targets:
            self._main_loop()

        return self.submitted

    # ── 统一提交入口 ────────────────────────────────────────

    def _submit(self, direction, code, shares):
        now = time.time()
        peer_price = now >= self.open_limit_deadline
        if direction == 'SELL':
            price = self.prices.get(code, 0)
            limit = None if peer_price else (
                self.executor._protect_price(price, self.executor.SELL_PROTECT_PCT, 'SELL')
                if price > 0 else None)
            r = self.executor.submit_order('SELL', code, shares, limit, self.signal_date, self.trade_date)
            otype = xtconstant.STOCK_SELL
        else:
            shares = floor_buy_shares(code, shares)
            if shares < min_buy_shares(code):
                return None
            price = self.prices.get(code, 0)
            limit = None if peer_price else (
                self.executor._protect_price(price, self.executor.BUY_PROTECT_PCT, 'BUY')
                if price > 0 else None)
            r = self.executor.submit_order('BUY', code, shares, limit, self.signal_date, self.trade_date)
            otype = xtconstant.STOCK_BUY

        peer_price = limit is None
        if not r:
            self.retry_after[code] = self._retry_at(
                self.executor.ORDER_REJECT_RETRY_SEC)
            self.record_action(f'{direction.lower()}_submit_failed', code=code,
                               order_type=otype, shares=shares)
            return None

        submitted_at = time.time()
        r['submitted_at'] = submitted_at
        r['direction'] = direction
        r['price_mode'] = 'peer' if peer_price else 'open'
        r['target_shares'] = (
            self.sell_targets.get(code, shares)
            if direction == 'SELL' else self.targets.get(code, shares))
        r['reprice_at'] = (
            self._peer_reprice_at(submitted_at)
            if peer_price else self.open_limit_deadline)
        self.submitted.append(r)
        self._submitted_by_id[r['order_id']] = r
        self.orders_by_code.setdefault(code, []).append(r['order_id'])
        self._local_inflight[r['order_id']] = shares
        self.reprice_pending.discard(code)
        amount = shares * self._unit_cost(code) if direction == 'BUY' else 0.0
        self.record_action(f'{direction.lower()}_submit', code=code, order_id=r['order_id'],
                           order_type=otype, shares=shares, amount=amount,
                           msg='peer_price' if peer_price else f'open_limit={limit:.2f}')
        return r

    def _retry_at(self, delay: float) -> float:
        now = time.time()
        retry_at = now + delay
        return min(retry_at, self.open_limit_deadline) if now < self.open_limit_deadline else retry_at

    def _peer_reprice_at(self, submitted_at: float) -> float:
        """对手价单按连续交易时间保留 30 分钟，跨午休时顺延。"""
        reprice_at = submitted_at + self.executor.PEER_ORDER_TTL_SEC
        lunch_start = datetime.combine(self.trade_date, dt_time(11, 30)).timestamp()
        lunch_end = datetime.combine(self.trade_date, dt_time(13, 0)).timestamp()
        if submitted_at < lunch_start <= reprice_at:
            reprice_at += lunch_end - lunch_start
        return reprice_at

    def _can_submit_now(self, now: float) -> bool:
        """09:31 前可挂 open 限价；其后只在连续竞价时段挂对手价。"""
        if now < self.open_limit_deadline:
            return True
        current = datetime.fromtimestamp(now)
        return (is_current_trading(current)
                and current.time() < self.executor.PEER_ORDER_END)

    def _can_cancel_now(self, now: float) -> bool:
        """收盘集合竞价前给异步撤单和重挂留出安全窗口。"""
        return (self._can_submit_now(now)
                and datetime.fromtimestamp(now).time() < self.executor.PEER_CANCEL_END)

    # ── 资金 / 单位成本 ──────────────────────────────────────

    def _available_cash(self) -> float:
        return self.executor._available_buy_cash(self.trader.query_asset())

    def _unit_cost(self, code) -> float:
        lp = self.limit_prices.get(code, 0)
        if lp and lp > 0:
            return lp * (1 + self.executor.BUY_FEE_RATE)
        return self.prices[code] * (1 + self.executor.BUY_FEE_RATE)

    # ── 订单追踪 ─────────────────────────────────────────────

    def _query_order(self, order_id):
        o = self.trader.query_order(order_id)
        if o:
            submitted = self._submitted_by_id.get(order_id)
            default_order_type = (
                xtconstant.STOCK_BUY
                if submitted and submitted['direction'] == 'BUY'
                else xtconstant.STOCK_SELL if submitted else 0)
            self._last_order_state[order_id] = (
                int(getattr(o, 'order_status', 0) or 0),
                int(getattr(o, 'order_volume', 0)
                    or self._local_inflight.get(order_id, 0)),
                int(getattr(o, 'traded_volume', 0) or 0),
                int(getattr(o, 'order_type', 0) or default_order_type),
                (getattr(o, 'status_msg', '') or '').strip(),
            )
            if submitted is not None:
                submitted['last_order_state'] = self._last_order_state[order_id]
        return o

    def _filled_inflight(self, code):
        filled = inflight = 0
        for oid in self.orders_by_code.get(code, []):
            self._query_order(oid)
            state = self._last_order_state.get(oid)
            if state is None:
                inflight += self._local_inflight.get(oid, 0)
                continue
            status, volume, traded, _, _ = state
            filled += traded
            if status not in TERMINAL_STATUS:
                inflight += max(0, volume - traded)
        return filled, inflight

    def _remaining(self, code):
        """买单剩余未成交股数"""
        filled, inflight = self._filled_inflight(code)
        return max(0, self.targets.get(code, 0) - filled - inflight)

    def _sell_remaining(self, code):
        """卖单剩余未成交股数"""
        filled, inflight = self._filled_inflight(code)
        return max(0, self.sell_targets.get(code, 0) - filled - inflight)

    def _has_open_order(self, code) -> bool:
        """该标的已有在途委托（含 QMT 尚未登记的新单）。"""
        _, inflight = self._filled_inflight(code)
        return inflight > 0

    def _has_pending_orders(self) -> bool:
        """任一买卖委托仍在途；新单尚未被 QMT 查询到时也视为在途。"""
        return any(self._has_open_order(code) for code in self.orders_by_code)

    def _has_unhandled_terminal(self, code) -> bool:
        return any(
            oid not in self.handled_terminal
            and (state := self._last_order_state.get(oid)) is not None
            and state[0] in TERMINAL_STATUS
            for oid in self.orders_by_code.get(code, [])
        )

    # ── 退出条件 ─────────────────────────────────────────────

    def _all_buys_done(self):
        for code in self.buy_seq:
            if self._remaining(code) >= min_buy_shares(code):
                return False
        return True

    def _all_targets_resolved(self):
        if self._has_pending_orders():
            return False
        if not self._all_buys_done():
            return False
        for code in self.sell_targets:
            if self._sell_remaining(code) > 0:
                return False
        return True

    def _can_afford_any(self, cash: float) -> bool:
        for code in self.buy_seq:
            minimum = min_buy_shares(code)
            if self._remaining(code) < minimum:
                continue
            if floor_buy_shares(code, int(cash / self._unit_cost(code))) >= minimum:
                return True
        return False

    def _waiting_for_retry(self) -> bool:
        now = time.time()
        for code in self.buy_seq:
            if self._remaining(code) < min_buy_shares(code):
                continue
            if self.retry_after.get(code, 0) > now:
                return True
        for code in self.sell_targets:
            if self._sell_remaining(code) <= 0:
                continue
            if self.retry_after.get(code, 0) > now:
                return True
        return False

    # ── 废单识别（买卖共用）──────────────────────────────────

    def _handle_terminal_orders(self):
        """检查所有订单（买+卖），0成交终态 → 设退避 timer"""
        now = time.time()
        for code, order_ids in self.orders_by_code.items():
            for oid in order_ids:
                if oid in self.handled_terminal:
                    continue
                self._query_order(oid)
                state = self._last_order_state.get(oid)
                if state is None or state[0] not in TERMINAL_STATUS:
                    continue
                _, volume, traded, order_type, msg = state
                self.handled_terminal.add(oid)
                if oid in self.cancel_requested:
                    self.retry_after.pop(code, None)
                    continue
                if traded > 0:
                    continue
                is_buy = order_type == xtconstant.STOCK_BUY
                if is_buy and _is_insufficient_funds(msg):
                    action = 'buy_underfunded'
                    delay = self.executor.UNDERFUNDED_BACKOFF_SEC
                else:
                    direction = 'buy' if is_buy else 'sell'
                    action = f'{direction}_reject_retry_later'
                    delay = self.executor.ORDER_REJECT_RETRY_SEC
                self.retry_after[code] = self._retry_at(delay)
                self.record_action(action, code=code, order_id=oid,
                                   order_type=order_type, shares=volume,
                                   msg=msg)

    # ── 买单提交 ─────────────────────────────────────────────

    def _submit_affordable_buys(self) -> bool:
        """按 TopN 顺序提交当前现金买得起的买单"""
        progressed = False
        now = time.time()
        if not self._can_submit_now(now):
            return False
        cash = self._available_cash()
        for code in self.buy_seq:
            if now < self.retry_after.get(code, 0):
                continue
            if (self._has_open_order(code)
                    or self._has_unhandled_terminal(code)):
                continue
            rem = self._remaining(code)
            if self._has_unhandled_terminal(code):
                continue
            min_lot = min_buy_shares(code)
            if rem < min_lot:
                continue
            afford = floor_buy_shares(code, int(cash / self._unit_cost(code)))
            shares = floor_buy_shares(code, min(rem, afford))
            if shares < min_lot:
                continue
            if self._submit('BUY', code, shares):
                cash -= shares * self._unit_cost(code)
                progressed = True
        return progressed

    # ── 卖单重试 ─────────────────────────────────────────────

    def _retry_sells(self) -> bool:
        """重试被拒卖单（超退避期后）"""
        progressed = False
        now = time.time()
        if not self._can_submit_now(now):
            return False
        for code in self.sell_targets:
            if now < self.retry_after.get(code, 0):
                continue
            if (self._has_open_order(code)
                    or self._has_unhandled_terminal(code)):
                continue
            rem = self._sell_remaining(code)
            if self._has_unhandled_terminal(code):
                continue
            if rem <= 0:
                continue
            pos = self.trader.query_stock_position(code)
            shares = min(
                rem, int(getattr(pos, 'can_use_volume', 0) or 0) if pos else 0)
            if shares > 0 and self._submit('SELL', code, shares):
                progressed = True
        return progressed

    # ── 到期撤单 ─────────────────────────────────────────────

    def _cancel_expired_orders(self, now: float) -> bool:
        """到点只发撤单请求；等旧单终态后由现有缺口逻辑重挂。"""
        if not self._can_cancel_now(now):
            return False
        progressed = False
        for s in self.submitted:
            oid = s['order_id']
            cancel_at = self.cancel_requested.get(oid)
            if (now < s['reprice_at']
                    or (cancel_at is not None
                        and now - cancel_at < self.executor.CANCEL_REQUEST_RETRY_SEC)):
                continue
            o = self._query_order(oid)
            if not o or o.order_status not in CANCELABLE_STATUS:
                continue
            remaining = max(
                0, int(getattr(o, 'order_volume', 0) or 0)
                - int(getattr(o, 'traded_volume', 0) or 0))
            if remaining <= 0:
                continue
            order_type = (xtconstant.STOCK_BUY if s['direction'] == 'BUY'
                          else xtconstant.STOCK_SELL)
            try:
                self.trader.cancel_order(oid)
            except Exception as e:
                if cancel_at is not None:
                    self.cancel_requested[oid] = now
                s['reprice_at'] = now + self.executor.ORDER_REJECT_RETRY_SEC
                self.record_action(
                    'reprice_cancel_failed', code=s['code'], order_id=oid,
                    order_type=order_type, shares=remaining, msg=str(e))
                continue
            self.cancel_requested[oid] = now
            self.reprice_pending.add(s['code'])
            s['intentional_cancel'] = True
            self.record_action(
                'reprice_cancel_requested', code=s['code'], order_id=oid,
                order_type=order_type, shares=remaining,
                msg=f"{s['price_mode']}->peer")
            progressed = True
        return progressed

    # ── 主循环 ───────────────────────────────────────────────

    def _monitor_deadline(self) -> float:
        override = getattr(self.executor, 'BUY_MONITOR_DEADLINE_SEC', None)
        if override is not None:
            return time.time() + float(override)
        return datetime.combine(self.trade_date, self.executor.BUY_MONITOR_END).timestamp()

    def _main_loop(self):
        deadline = self._monitor_deadline()
        last_log = time.time()
        while time.time() < deadline:
            self._handle_terminal_orders()
            reprice_progress = self._cancel_expired_orders(time.time())
            if time.time() - last_log >= 30:
                trading_logger.info(
                    f"[OrderMonitor] 监控中: 已提交 {len(self.submitted)} 笔, "
                    f"委托在途={self._has_pending_orders()}")
                last_log = time.time()

            buy_progress = self._submit_affordable_buys()
            sell_progress = self._retry_sells()
            progressed = reprice_progress or buy_progress or sell_progress

            if self._all_targets_resolved():
                return

            if not progressed:
                if self._waiting_for_retry():
                    time.sleep(self.executor.MONITOR_POLL_SEC)
                    continue
                cash = self._available_cash()
                sells_left = any(
                    self._sell_remaining(code) > 0 for code in self.sell_targets)
                if (not self._has_pending_orders() and not self._can_afford_any(cash)
                        and not sells_left and not self.reprice_pending):
                    trading_logger.warning(
                        f"[OrderMonitor] 可用资金 {cash:.2f} 不足以买入任何剩余标的, 结束")
                    return
                time.sleep(self.executor.MONITOR_POLL_SEC)

    # ── 日志 / 落盘 ─────────────────────────────────────────

    def record_action(self, action: str, *, code: str = '', order_id: int = 0,
                      order_type: int | None = None, shares: int = 0,
                      amount: float = 0.0, msg: str = ''):
        trading_logger.debug(
            f"[ExecutorAction] action={action} code={code} order_id={order_id} "
            f"order_type={order_type} shares={shares} amount={amount:.2f} msg={msg}"
        )
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


class RebalanceExecutor:
    """调仓下单执行器,依赖一个 `Trader` 实例进行实际委托。"""

    BUY_FEE_RATE = LIVE_BUY_FEE_RATE       # 实盘券商冻结费率(佣金+过户费)
    BUY_PROTECT_PCT = 0.0     # 买入限价 = 开盘价（限价 ≤ 开盘价才能成交）
    SELL_PROTECT_PCT = 0.0    # 卖出限价 = 开盘价（限价 ≤ 开盘价才能成交）
    OPEN_LIMIT_END = dt_time(9, 31)
    PEER_ORDER_TTL_SEC = 30 * 60
    CANCEL_REQUEST_RETRY_SEC = 5.0
    PEER_CANCEL_END = dt_time(14, 56)  # 给异步撤单终态和重挂预留 1 分钟
    PEER_ORDER_END = dt_time(14, 57)  # 收盘集合竞价不可撤单，不再换价
    BUY_MONITOR_END = dt_time(15, 0)
    OPEN_LIMIT_DEADLINE_SEC = None    # 测试可覆盖首次切换为相对秒数
    BUY_MONITOR_DEADLINE_SEC = None    # 测试/闭市演练可覆盖为秒级
    UNDERFUNDED_BACKOFF_SEC = 3.0      # 资金不足废单后退避,等 QMT cash 更新
    ORDER_REJECT_RETRY_SEC = 60.0      # 价格范围/柜台废单后按原价低频重试
    SETTLE_WAIT_SEC = 5                # 提交结束后短暂沉淀回调;不撤未终态订单
    MONITOR_POLL_SEC = 0.3             # 买入资金门控轮询间隔
    # --skip 且真实时钟在闭市: order_stock 同步 API 可阻塞数分钟(非「废单慢」);
    # 此时跳过真实委托,仅收口战报/盘后。盘中委托仍走正常轮询超时。
    _TIMEOUT_FIELDS = (
        'OPEN_LIMIT_DEADLINE_SEC', 'BUY_MONITOR_DEADLINE_SEC', 'SETTLE_WAIT_SEC',
    )
    _OFF_HOURS_TIMEOUTS = {
        'BUY_MONITOR_DEADLINE_SEC': 0.5,
        'SETTLE_WAIT_SEC': 10,
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
        """QMT 买入校验使用 XtAsset.cash（可用金额,实时含冻结扣减与卖出回款）。

        2026-06-11 复盘:current_balance/fetch_balance 是「上日结存/可取金额」口径,
        盘中恒等于 T-1 日终现金、不反映当日卖出回款 → 本地资金预检全天失真:
        只买得起最便宜的一只票(全部资金追入秦川物联),扬州金泉/美腾科技一笔未发。
        禁止再使用这两个字段做买入校验。
        """
        if not asset:
            return 0.0
        return max(0.0, float(getattr(asset, 'cash', 0.0) or 0.0))

    @staticmethod
    def _protect_price(price: float, pct: float, direction: str) -> float:
        """保护限价:price×(1±pct),按 A 股最小报价单位取整。direction='BUY'上浮,'SELL'下浮。"""
        sign = 1 if direction == 'BUY' else -1
        tick = 0.001 if price < 1.0 else 0.01
        raw = price * (1 + sign * pct)
        return round(round(raw / tick) * tick, 3 if tick == 0.001 else 2)

    def submit_order(self, direction, code, shares, limit_price, signal_date, trade_date):
        """统一的 QMT 下单入口。direction='BUY'|'SELL',返回 {code,order_type,order_id,shares} 或 None。"""
        order_type = xtconstant.STOCK_BUY if direction == 'BUY' else xtconstant.STOCK_SELL
        remark = f'rebalance signal={signal_date.isoformat()} trade={trade_date.isoformat()}'
        order_id = self.trader.order(order_type, code, shares, limit_price, order_remark=remark)
        if order_id is None:
            trading_logger.info(f"{code} {direction}委托未发出")
            return None
        px_msg = f' @{limit_price:.2f}' if limit_price is not None else ' @对手价'
        trading_logger.info(f"已提交{direction}委托: {code} {shares}股{px_msg} order_id={order_id}")
        recorder.mark(f"提交{direction}委托")
        return {'code': code, 'order_type': direction, 'order_id': order_id, 'shares': shares}

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
        coverage = {}
        for s in submitted:
            o = self.trader.query_order(s['order_id'])
            state = s.get('last_order_state')
            if o:
                order_status = int(getattr(o, 'order_status', 0) or 0)
                vol = int(getattr(o, 'order_volume', 0) or 0)
                traded = int(getattr(o, 'traded_volume', 0) or 0)
                msg = getattr(o, 'status_msg', '') or ''
            elif state:
                order_status, vol, traded, _, msg = state
            else:
                order_status, vol, traded, msg = None, int(s['shares']), 0, ''

            target = s.get('target_shares')
            if target is not None:
                key = (s['order_type'], s['code'])
                item = coverage.setdefault(
                    key, {'target': int(target), 'filled': 0, 'inflight': 0})
                item['target'] = max(item['target'], int(target))
                item['filled'] += traded
                if order_status not in TERMINAL_STATUS:
                    item['inflight'] += max(0, vol - traded)

            if (s.get('intentional_cancel')
                    and order_status in (xtconstant.ORDER_CANCELED,
                                         xtconstant.ORDER_PART_CANCEL)):
                continue
            status = (
                get_order_status_label(order_status)
                if order_status is not None else '查询失败')
            price = o.traded_price if o else 0
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

        for (direction, code), item in coverage.items():
            missing = max(
                0, item['target'] - item['filled'] - item['inflight'])
            if missing <= 0:
                continue
            detail = get_stock_detail(code)
            name = (detail.get('InstrumentName', '') if detail else '').strip()
            label = f"{code} {name}".strip()
            msg = (
                f"target={item['target']} filled={item['filled']} "
                f"inflight={item['inflight']}")
            fail_rows.append({
                'order_type': direction,
                'code': code,
                'name': name,
                'shares': missing,
                'status': '目标未覆盖',
                'msg': msg,
                'line': f"{direction:4s} {label} {missing}股 → 目标未覆盖 {msg}",
            })

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
        for code, shares in sell_orders:
            trading_logger.info(
                f"[执行计划-卖] code={code} shares={shares} "
                f"close_price={prices.get(code, 0):.4f}")
        for code in buy_n_stocks:
            if code not in buy_allocations:
                continue
            trading_logger.info(
                f"[执行计划-买] code={code} shares={int(buy_allocations[code])} "
                f"close_price={prices.get(code, 0):.4f} "
                f"freeze_price={limit_prices.get(code, 0):.4f}")

        if getattr(self, '_skip_real_orders', False):
            self._summarize([])
            return []

        avail = self._available_buy_cash(self.trader.query_asset())
        live_trade_recorder.record_event(
            EVT_EXECUTION_ACTION,
            source=SRC_EXECUTOR,
            trade_date=trade_date,
            status_msg=f"rebalance_start: available_cash={avail:.2f}",
        )

        monitor = OrderMonitor(
            self,
            sell_orders=sell_orders if execute_sell else [],
            buy_allocations=buy_allocations if execute_buy else {},
            buy_n_stocks=buy_n_stocks,
            prices=prices,
            signal_date=signal_date,
            trade_date=trade_date,
            limit_prices=limit_prices,
        )
        submitted = monitor.run()
        self._wait_terminal(submitted)
        self._summarize(submitted)
        return submitted
