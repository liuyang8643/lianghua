"""Shared execution state machine. Python 3.6; no strategy/dataframe imports.

Broker operations and audit sinks are supplied by the application adapter.
The local compatibility adapter remains in trading.executor.
"""
import time
import math
import logging
from datetime import datetime, time as dt_time
from xtquant import xtconstant
from core.fees import LIVE_BUY_FEE_RATE
from trading.helper import get_order_status_label
from utils.stock.info import floor_buy_shares, floor_sell_shares, min_buy_shares
from trading.target_protocol import TERMINAL_ORDER_STATUSES

EVT_EXECUTION_ACTION = 'execution_action'
SRC_EXECUTOR = 'executor_monitor'

TERMINAL_STATUS = TERMINAL_ORDER_STATUSES


CANCELABLE_STATUS = {
    xtconstant.ORDER_REPORTED, xtconstant.ORDER_PART_SUCC,
}


QMT_INSUFFICIENT_FUNDS_CODE = '150906130'


def _is_insufficient_funds(msg) :
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
        self.submitted = []

        # 解析卖单：负数 → 查 QMT 取实际可卖股数
        self.sell_targets = {}
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
        self.orders_by_code = {}
        self.handled_terminal = set()
        self.retry_after = {}
        self._local_inflight = {}
        self._last_order_state = {}
        self._submitted_by_id = {}
        self.cancel_requested = {}
        self.reprice_pending = set()
        self.sell_cash_settle_until = 0.0
        override = getattr(self.executor, 'OPEN_LIMIT_DEADLINE_SEC', None)
        now = time.time()
        market_open = datetime.combine(
            self.trade_date, self.executor.MARKET_OPEN).timestamp()
        scheduled_deadline = datetime.combine(
            self.trade_date, self.executor.OPEN_LIMIT_END).timestamp()
        self.open_limit_deadline = (
            now + float(override) if override is not None
            else (scheduled_deadline if now < market_open
                  else now + self.executor.OPEN_LIMIT_TTL_SEC)
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
                shares = self._sell_submission_quantity(code, rem)
                if shares > 0:
                    self._submit('SELL', code, shares)

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

    def _retry_at(self, delay) :
        now = time.time()
        retry_at = now + delay
        return min(retry_at, self.open_limit_deadline) if now < self.open_limit_deadline else retry_at

    def _peer_reprice_at(self, submitted_at) :
        """对手价单按连续交易时间保留 30 秒，跨午休时顺延。"""
        reprice_at = submitted_at + self.executor.PEER_ORDER_TTL_SEC
        lunch_start = datetime.combine(self.trade_date, dt_time(11, 30)).timestamp()
        lunch_end = datetime.combine(self.trade_date, dt_time(13, 0)).timestamp()
        if submitted_at < lunch_start <= reprice_at:
            reprice_at += lunch_end - lunch_start
        return reprice_at

    def _can_submit_now(self, now) :
        """启动后 60 秒挂 open 限价；其后只在连续竞价时段挂对手价。"""
        current = datetime.fromtimestamp(now)
        if current.date() != self.trade_date or not self.executor._is_current_trading(current):
            return False
        if now < self.open_limit_deadline:
            return True
        return current.time() < self.executor.PEER_ORDER_END

    def _can_cancel_now(self, now) :
        """收盘集合竞价前给异步撤单和重挂留出安全窗口。"""
        return (self._can_submit_now(now)
                and datetime.fromtimestamp(now).time() < self.executor.PEER_CANCEL_END)

    # ── 资金 / 单位成本 ──────────────────────────────────────

    def _available_cash(self) :
        asset = self.trader.query_asset()
        if asset is None or asset.cash is None or not math.isfinite(float(asset.cash)) or float(asset.cash) < 0:
            return None
        return self.executor._available_buy_cash(asset)

    def _unit_cost(self, code) :
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

    def _has_open_order(self, code) :
        """该标的已有在途委托（含 QMT 尚未登记的新单）。"""
        _, inflight = self._filled_inflight(code)
        return inflight > 0

    def _has_pending_orders(self) :
        """任一买卖委托仍在途；新单尚未被 QMT 查询到时也视为在途。"""
        return any(self._has_open_order(code) for code in self.orders_by_code)

    def _sells_pending(self) :
        """兼容旧调用：是否仍有卖单在途。"""
        return any(self._has_open_order(code) for code in self.sell_targets)

    def _has_unhandled_terminal(self, code) :
        for oid in self.orders_by_code.get(code, []):
            state = self._last_order_state.get(oid)
            if oid not in self.handled_terminal and state is not None and state[0] in TERMINAL_STATUS:
                return True
        return False

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

    def _can_afford_any(self, cash) :
        for code in self.buy_seq:
            minimum = min_buy_shares(code)
            if self._remaining(code) < minimum:
                continue
            if floor_buy_shares(code, int(cash / self._unit_cost(code))) >= minimum:
                return True
        return False

    def _waiting_for_retry(self) :
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
                is_buy = order_type == xtconstant.STOCK_BUY
                if traded > 0 and not is_buy:
                    self.sell_cash_settle_until = max(
                        self.sell_cash_settle_until,
                        now + self.executor.SELL_CASH_SETTLE_WAIT_SEC,
                    )
                if oid in self.cancel_requested:
                    self.retry_after.pop(code, None)
                    continue
                if traded > 0:
                    continue
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

    def _submit_affordable_buys(self) :
        """按 TopN 顺序提交当前现金买得起的买单"""
        progressed = False
        now = time.time()
        if not self._can_submit_now(now):
            return False
        cash = self._available_cash()
        if cash is None:
            return False
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

    def _retry_sells(self) :
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
            shares = self._sell_submission_quantity(code, rem)
            if shares > 0 and self._submit('SELL', code, shares):
                progressed = True
        return progressed

    def _sell_submission_quantity(self, code, remaining):
        if not hasattr(self.trader, 'query_stock_position'):
            return remaining
        position = self.trader.query_stock_position(code)
        return (floor_sell_shares(code, remaining, position.can_use_volume, position.volume)
                if position else 0)

    # ── 到期撤单 ─────────────────────────────────────────────

    def _cancel_expired_orders(self, now) :
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
            if s['direction'] == 'BUY':
                can_replace = floor_buy_shares(s['code'], remaining) == remaining
            else:
                position = self.trader.query_stock_position(s['code'])
                can_replace = bool(position and floor_sell_shares(
                    s['code'], remaining, remaining, position.volume) == remaining)
            if not can_replace:
                # The accepted order may keep filling below a new-order lot
                # minimum. Cancelling it would strand the exact target balance.
                s['reprice_at'] = now + self.executor.PEER_ORDER_TTL_SEC
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

    def _monitor_deadline(self) :
        override = getattr(self.executor, 'BUY_MONITOR_DEADLINE_SEC', None)
        if override is not None:
            return time.time() + float(override)
        return datetime.combine(self.trade_date, self.executor.BUY_MONITOR_END).timestamp()

    def _main_loop(self):
        deadline = self._monitor_deadline()
        last_log = 0.0
        last_state = None
        while time.time() < deadline:
            self.executor._on_tick()
            now = time.time()
            self._handle_terminal_orders()
            reprice_progress = self._cancel_expired_orders(now)
            pending = self._has_pending_orders()
            can_submit = self._can_submit_now(now)
            state = (len(self.submitted), pending, can_submit)
            if state != last_state or now - last_log >= 300:
                if not can_submit and not pending:
                    self.executor.logger.info("[OrderMonitor] 等待连续竞价时段")
                else:
                    self.executor.logger.info(
                        f"[OrderMonitor] 已提交 {len(self.submitted)} 笔, 委托在途={pending}")
                last_state = state
                last_log = now

            sell_progress = self._retry_sells()
            buy_progress = self._submit_affordable_buys()
            progressed = reprice_progress or buy_progress or sell_progress

            if self._all_targets_resolved():
                return

            if not progressed:
                if self._waiting_for_retry():
                    time.sleep(self.executor.MONITOR_POLL_SEC)
                    continue
                # 09:30 前及午休不需要查询可用资金；等待交易窗口即可。
                if not self._can_submit_now(time.time()):
                    time.sleep(self.executor.MONITOR_POLL_SEC)
                    continue
                cash = self._available_cash()
                if cash is None:
                    time.sleep(self.executor.MONITOR_POLL_SEC)
                    continue
                sells_left = any(
                    self._sell_remaining(code) > 0 for code in self.sell_targets)
                if (not self._has_pending_orders() and not self._can_afford_any(cash)
                        and not sells_left and not self.reprice_pending):
                    if time.time() < self.sell_cash_settle_until:
                        time.sleep(self.executor.MONITOR_POLL_SEC)
                        continue
                    if self.executor.journal is not None:
                        # 可用现金可能稍后解冻；实盘保留原目标直到收盘，不把零委托冒充完成。
                        time.sleep(self.executor.MONITOR_POLL_SEC)
                        continue
                    self.executor.logger.warning(
                        f"[OrderMonitor] 可用资金 {cash:.2f} 不足以买入任何剩余标的, 结束")
                    return
                time.sleep(self.executor.MONITOR_POLL_SEC)

    # ── 日志 / 落盘 ─────────────────────────────────────────

    def record_action(self, action, *, code = '', order_id = 0,
                      order_type = None, shares = 0,
                      amount = 0.0, msg = ''):
        self.executor.logger.debug(
            f"[ExecutorAction] action={action} code={code} order_id={order_id} "
            f"order_type={order_type} shares={shares} amount={amount:.2f} msg={msg}"
        )
        try:
            self.executor._record_event(
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
        except Exception as exc:
            # 券商已接受委托后，本地审计落盘失败绝不能中止后续下单。
            self.executor.logger.error(
                f"[交易审计] {action} 落盘失败，交易继续: {exc}"
            )


class RebalanceExecutor:
    """调仓下单执行器,依赖一个 `Trader` 实例进行实际委托。"""

    BUY_FEE_RATE = LIVE_BUY_FEE_RATE       # 实盘券商冻结费率(佣金+过户费)
    BUY_PROTECT_PCT = 0.0     # 买入限价 = 开盘价（限价 ≤ 开盘价才能成交）
    SELL_PROTECT_PCT = 0.0    # 卖出限价 = 开盘价（限价 ≤ 开盘价才能成交）
    MARKET_OPEN = dt_time(9, 30)
    OPEN_LIMIT_END = dt_time(9, 31)
    OPEN_LIMIT_TTL_SEC = 60
    PEER_ORDER_TTL_SEC = 30
    CANCEL_REQUEST_RETRY_SEC = 5.0
    PEER_CANCEL_END = dt_time(14, 56)  # 给异步撤单终态和重挂预留 1 分钟
    PEER_ORDER_END = dt_time(14, 57)  # 收盘集合竞价不可撤单，不再换价
    BUY_MONITOR_END = dt_time(15, 0)
    OPEN_LIMIT_DEADLINE_SEC = None    # 测试可覆盖首次切换为相对秒数
    BUY_MONITOR_DEADLINE_SEC = None    # 测试/闭市演练可覆盖为秒级
    UNDERFUNDED_BACKOFF_SEC = 3.0      # 资金不足废单后退避,等 QMT cash 更新
    ORDER_REJECT_RETRY_SEC = 60.0      # 价格范围/柜台废单后按原价低频重试
    SELL_CASH_SETTLE_WAIT_SEC = 10.0   # 卖单终态先于 QMT cash 刷新时的有限等待
    SETTLE_WAIT_SEC = 5                # 提交结束后短暂沉淀回调;不撤未终态订单
    MONITOR_POLL_SEC = 1.0             # 买入资金门控轮询间隔
    # --skip 且真实时钟在闭市: order_stock 同步 API 可阻塞数分钟(非「废单慢」);
    # 此时跳过真实委托,仅收口战报/盘后。盘中委托仍走正常轮询超时。
    _TIMEOUT_FIELDS = (
        'OPEN_LIMIT_DEADLINE_SEC', 'BUY_MONITOR_DEADLINE_SEC', 'SETTLE_WAIT_SEC',
    )
    _OFF_HOURS_TIMEOUTS = {
        'BUY_MONITOR_DEADLINE_SEC': 0.5,
        'SETTLE_WAIT_SEC': 10,
    }

    def __init__(self, trader, *, journal=None):
        self.trader = trader
        self.journal = journal
        self._monitor = None

    def _snapshot_timeouts(self) :
        return {k: getattr(self, k) for k in self._TIMEOUT_FIELDS}

    def _restore_timeouts(self, snap):
        for k, v in snap.items():
            setattr(self, k, v)

    def _apply_off_hours_timeouts(self):
        for k, v in self._OFF_HOURS_TIMEOUTS.items():
            setattr(self, k, v)

    @staticmethod
    def _available_buy_cash(asset) :
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
    def _protect_price(price, pct, direction) :
        """保护限价:price×(1±pct),按 A 股最小报价单位取整。direction='BUY'上浮,'SELL'下浮。"""
        sign = 1 if direction == 'BUY' else -1
        tick = 0.001 if price < 1.0 else 0.01
        raw = price * (1 + sign * pct)
        return round(round(raw / tick) * tick, 3 if tick == 0.001 else 2)

    def submit_order(self, direction, code, shares, limit_price, signal_date, trade_date):
        """统一的 QMT 下单入口。direction='BUY'|'SELL',返回 {code,order_type,order_id,shares} 或 None。"""
        order_type = xtconstant.STOCK_BUY if direction == 'BUY' else xtconstant.STOCK_SELL
        remark = f'rebalance signal={signal_date.isoformat()} trade={trade_date.isoformat()}'
        if self.journal is not None:
            request = self.journal.before_submit(direction, code, shares, limit_price)
            remark = request['remark']
        from trading.bridge_wire import ConfirmedNotSubmitted, BridgeUnavailable
        try:
            order_id = self.trader.order(order_type, code, shares, limit_price, order_remark=remark)
        except ConfirmedNotSubmitted as exc:
            if self.journal is not None:
                self.journal.mark_not_submitted(request, exc)
            self.logger.warning(f'{code} {direction}确认未提交，保留原目标等待重试: {exc}')
            return None
        except BridgeUnavailable as exc:
            if exc.not_submitted and self.journal is not None:
                self.journal.mark_not_submitted(request, '客户端心跳校验失败，请求尚未进入执行队列')
            raise
        if order_id is None:
            if self.journal is not None:
                raise RuntimeError('下单返回为空，必须先向券商核实执行记录，禁止直接重发')
            self.logger.info(f"{code} {direction}委托未发出")
            return None
        px_msg = f' @{limit_price:.2f}' if limit_price is not None else ' @对手价'
        self.logger.info(f"已提交{direction}委托: {code} {shares}股{px_msg} order_id={order_id}")
        self._mark(f"提交{direction}委托")
        return {'code': code, 'order_type': direction, 'order_id': order_id, 'shares': shares}

    def _wait_terminal(self, submitted):
        """轮询等待所有委托进入终态。"""
        waited = 0
        pending_orders = []
        while waited < self.SETTLE_WAIT_SEC:
            pending_orders = []
            for submitted_order in submitted:
                order = self.trader.query_order(submitted_order['order_id'])
                if order and order.order_status not in TERMINAL_STATUS:
                    pending_orders.append(submitted_order['code'])

            if not pending_orders:
                return
            time.sleep(1)
            waited += 1
        self.logger.warning(f"等待成交超时, 仍有{len(pending_orders)}笔未终态: {pending_orders[:5]}...")

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
            detail = self._stock_detail(s['code'])
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
            detail = self._stock_detail(code)
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
            self.logger.info(f"[调仓汇总] 成交 {len(ok_list)} 笔")
            for l in ok_list:
                self.logger.debug(f"  {l}")
        if partial_list:
            self.logger.warning(f"=== 未完成 {len(partial_list)} 笔 ===")
            for l in partial_list:
                self.logger.warning(f"  {l}")
        if fail_rows:
            self.logger.warning(f"=== 失败 {len(fail_rows)} 笔 ===")
            for r in fail_rows:
                self.logger.warning(f"  {r['line']}")

    def execute(self, pending, *, execute_sell=True, execute_buy=True,
                off_hours_fast = False):
        """执行一次调仓:提交卖买委托、等待终态、汇总。

        Args:
            pending: before_trade 产出的 pending_rebalance 字典。
            execute_sell / execute_buy: 是否执行卖 / 买（分时段调仓时使用）。
            off_hours_fast: 真实时钟在闭市且 --skip 演练时缩短轮询超时,避免卡 10min+。
        """
        timeout_snap = self._snapshot_timeouts()
        self._skip_real_orders = off_hours_fast
        if off_hours_fast:
            self.logger.warning(
                "[执行] 闭市 --skip: QMT order_stock 同步调用会长时间阻塞(实测>3min), "
                "非「秒废单」; 跳过真实买卖, 仅战报定稿+盘后 diff")
        try:
            return self._execute_impl(
                pending, execute_sell=execute_sell, execute_buy=execute_buy)
        except Exception:
            # 下单返回前进程/连接可能出错。重新进入时必须查询券商并恢复意图，不能重发原目标。
            if self.journal is not None:
                self._monitor = None
            raise
        finally:
            self._skip_real_orders = False
            self._restore_timeouts(timeout_snap)

    def _execute_impl(self, pending, *, execute_sell=True, execute_buy=True):
        if self.journal is not None and not getattr(self, '_skip_real_orders', False):
            if self.journal.load(pending['trade_date']) is None:
                orders = self.trader.query_orders()
                if any(o.strategy_name == self.trader.strategy_name for o in orders):
                    raise RuntimeError('当日已有本策略委托但缺执行记录，禁止重复执行；需先对账')
                # 清仓目标必须在第一笔下单前固定，不能在部分成交重启后重算并再减成交量。
                pending = dict(pending)
                if any(shares < 0 for _, shares in pending.get('sell_orders', [])):
                    _, positions = self.trader.query_account_snapshot()
                    available = {p.stock_code: int(p.can_use_volume) for p in positions}
                    pending['sell_orders'] = [(code, available.get(code, 0) if shares < 0 else shares)
                                              for code, shares in pending['sell_orders']]
            pending = self.journal.begin(pending)
            if self.journal.state['completed']:
                return []
        signal_date = pending['signal_date']
        trade_date = pending['trade_date']
        sell_orders = pending.get('sell_orders', [])
        buy_allocations = pending.get('buy_allocations', {})
        buy_n_stocks = pending.get('buy_n_stocks', list(buy_allocations.keys()))
        prices = pending.get('prices', {})
        limit_prices = pending.get('limit_prices', {})

        self.logger.info(
            f"开始调仓: sell={len(sell_orders)} buy={len(buy_allocations)} "
            f"signal={signal_date.isoformat()} trade={trade_date.isoformat()}")
        for code, shares in sell_orders:
            self.logger.debug(
                f"[执行计划-卖] code={code} shares={shares} "
                f"open_price={prices.get(code, 0):.4f}")
        for code in buy_n_stocks:
            if code not in buy_allocations:
                continue
            self.logger.debug(
                f"[执行计划-买] code={code} shares={int(buy_allocations[code])} "
                f"open_price={prices.get(code, 0):.4f} "
                f"freeze_price={limit_prices.get(code, 0):.4f}")

        if getattr(self, '_skip_real_orders', False):
            self._summarize([])
            return []

        avail = self._available_buy_cash(self.trader.query_asset())
        try:
            self._record_event(
                EVT_EXECUTION_ACTION,
                source=SRC_EXECUTOR,
                trade_date=trade_date,
                status_msg=f"rebalance_start: available_cash={avail:.2f}",
            )
        except Exception as exc:
            self.logger.error(
                f"[交易审计] rebalance_start 落盘失败，交易继续: {exc}"
            )

        monitor = self._monitor
        if monitor is None or monitor.trade_date != trade_date:
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
            if self.journal is not None:
                if 'open_limit_deadline' not in self.journal.state:
                    self.journal.state['open_limit_deadline'] = monitor.open_limit_deadline
                    self.journal.save()
                monitor.open_limit_deadline = self.journal.state['open_limit_deadline']
                proof_query = getattr(self.trader, 'query_not_submitted_intents', None)
                not_submitted = (proof_query(self.journal.state['requests'])
                                 if proof_query is not None else ())
                for request, order in self.journal.recover_orders(
                        self.trader.query_orders(), not_submitted=not_submitted):
                    direction = request['direction']
                    code = request['code']
                    oid = int(order.order_id)
                    record = {'code': code, 'shares': request['shares'], 'order_id': oid,
                              'order_type': direction, 'direction': direction,
                              'target_shares': (monitor.sell_targets.get(code, 0) if direction == 'SELL'
                                                else monitor.targets.get(code, 0)),
                              'price_mode': 'open' if request['price'] is not None else 'peer',
                              'reprice_at': monitor.open_limit_deadline if request['price'] is not None else time.time()}
                    monitor.submitted.append(record)
                    monitor._submitted_by_id[oid] = record
                    monitor.orders_by_code.setdefault(code, []).append(oid)
                    monitor._local_inflight[oid] = request['shares']
                    monitor._query_order(oid)
            self._monitor = monitor
        submitted = monitor.run()
        self._wait_terminal(submitted)
        self._summarize(submitted)
        if self.journal is not None:
            self.journal.state['completed'] = monitor._all_targets_resolved()
            self.journal.save()
        return submitted

    @property
    def logger(self):
        return logging.getLogger('wbr.execution')

    def _is_current_trading(self, now):
        t = now.time()
        return dt_time(9, 30) <= t < dt_time(11, 30) or dt_time(13) <= t < dt_time(15)

    def _record_event(self, *args, **kwargs):
        raise NotImplementedError('Execution audit sink is required')

    def _stock_detail(self, code):
        return {}

    def _mark(self, message):
        pass

    def _on_tick(self):
        pass

