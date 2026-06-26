"""调仓下单执行器。

把 `before_trade` 算好的 `pending_rebalance` 计划落到 QMT 委托,是实盘买卖动作的唯一入口。

执行模型(因子算完立即挂单 + 事后可复盘):

1. 输入计划
   - `pending['sell_orders']`: 需卖出的 `(code, shares)`,`shares < 0` 表示清仓。
   - `pending['buy_allocations']`: 每只目标买入股数(已含 reserve)。
   - `pending['buy_n_stocks']`: 买入优先级顺序(topN)。
   - `pending['prices']`: 盘前开盘估算价。
   - `pending['limit_prices']`: 每只涨停价(前收×(1+板块涨跌幅)),买单按它估资金冻结。

2. 卖、买挂单模型
   - 卖单:一次性提交全部 open[T] 限价单,不撤单、不重挂,让订单留在交易所队列里。
   - 买单:按 TopN 顺序检查 QMT 可用资金,够一只/一部分就立即提交 open[T] 限价单。
   - 资金暂时不够:不下废单,轮询 QMT 可用资金;任何卖出成交回款到账后继续挂后续买单。

3. 买单:持久挂单、不撤单
   - 已提交买单不因 30s 未成交撤单,保留时间优先权。
   - 资金不足废单只等待回款后重试,不熔断。
   - 非资金类废单(价格范围/柜台限制)不减手、不熔断,按同一 open[T] 限价低频重试到收盘前。
   - 无需本地账本:每次提交前重新查询 QMT 可用资金;在途冻结由 QMT cash 反映。

4. 资金口径
   - 买入校验用「可用资金」,`_available_buy_cash()` 优先 `current_balance`/`fetch_balance`,
     再回退 `cash - frozen_cash`。
   - 单笔占用按涨停价估(券商市价单按涨停价冻结),避免按开盘价低估 → 「资金可用数不足」废单。

5. 飞书与本地留存
   - QMT 原始回调由 `watcher.py` 落 events;executor 的提交/资金等待/废单重试落 `execution_action`。
   - 失败委托聚合成一张飞书卡(`_send_failure_summary_card`),不逐条 ERROR 刷屏。

6. 收尾:提交阶段汇总;未终态订单继续挂在柜台/交易所,后续回调由 watcher 记录。
"""
import time
from datetime import datetime, time as dt_time


from xtquant import xtconstant

from core.fees import LIVE_BUY_FEE_RATE, LIVE_BUY_PRICE_BUFFER
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

TERMINAL_STATUS = {
    xtconstant.ORDER_SUCCEEDED, xtconstant.ORDER_CANCELED,
    xtconstant.ORDER_JUNK, xtconstant.ORDER_PART_CANCEL,
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
        trading_logger.info(
            f"[ExecutorAction] action={action} code={code} order_id={order_id} "
            f"order_type={order_type} shares={shares} amount={amount:.2f} msg={msg}"
        )
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
    """卖单 monitor:一次性提交卖单,供买单侧判断卖出回款是否仍在途。"""

    def __init__(self, executor, sell_orders, signal_date, trade_date, prices=None):
        super().__init__(executor, signal_date, trade_date)
        self.sell_orders = list(sell_orders or [])
        self.prices = prices or {}

    def all_terminal(self) -> bool:
        """全部卖单是否已达终态（即回款不再有新增在途）。"""
        for s in self.submitted:
            o = self.trader.query_order(s['order_id'])
            if o and o.order_status not in TERMINAL_STATUS:
                return False
        return True

    def _submit_initial_sells(self):
        """串行提交全部卖单并登记到 self.submitted,供买单资金门控判断卖单是否仍在途。"""
        for code, shares in self.sell_orders:
            open_px = self.prices.get(code, 0)
            limit_price = (self.executor._sell_protect_price(open_px, self.executor.SELL_PROTECT_PCT)
                           if open_px > 0 else None)
            r = self.executor._submit_sell_order(code, shares,
                                                  self.signal_date, self.trade_date,
                                                  limit_price=limit_price)
            if r:
                r['submitted_at'] = time.time()
                self.submitted.append(r)
                self.record_action(
                    'sell_submit',
                    code=r['code'],
                    order_id=r['order_id'],
                    order_type=xtconstant.STOCK_SELL,
                    shares=r['shares'],
                )
        return self.submitted

class BuyMonitor(BaseMonitor):
    """买单 monitor:资金够就挂 open[T] 限价单,挂上后不撤单。

    只做两件事:
      1. 用 QMT cash 判断当前还能挂多少买单;
      2. 卖单回款或废单冷却后,继续按 TopN 顺序挂剩余买单。
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
        self.handled_terminal_orders: set[int] = set()
        # 废单后低频重试,避免价格范围暂不可报时刷爆柜台。
        self.retry_after: dict[str, float] = {}
        # 并发卖买时由 execute() 注入,用于判断卖单回款是否仍在途
        self.sell_monitor: SellMonitor | None = None

    def run(self):
        if not self.buy_seq:
            return []
        self.record_action('buy_monitor_start', msg=f"{len(self.buy_seq)} targets")
        self._buy_loop()
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
        """卖单回款是否仍在途——实时查卖单终态,不依赖外部标志位。"""
        sm = self.sell_monitor
        return bool(sm is not None and not sm.all_terminal())

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
            if self._remaining(code) >= min_buy_shares(code):
                return False
        return True

    def _can_afford_any(self, cash: float) -> bool:
        """可用现金是否买得起任意剩余标的最小1手。"""
        for code in self.buy_seq:
            minimum = min_buy_shares(code)
            if self._remaining(code) < minimum:
                continue
            if floor_buy_shares(code, int(cash / self._unit_cost(code))) >= minimum:
                return True
        return False

    def _monitor_deadline(self) -> float:
        override = getattr(self.executor, 'BUY_MONITOR_DEADLINE_SEC', None)
        if override is not None:
            return time.time() + float(override)
        return datetime.combine(self.trade_date, self.executor.BUY_MONITOR_END).timestamp()

    def _handle_terminal_orders(self):
        """处理已终态零成交委托:资金不足等回款,其他柜台废单按原价低频重试。"""
        now = time.time()
        for code, order_ids in self.orders_by_code.items():
            for oid in order_ids:
                if oid in self.handled_terminal_orders:
                    continue
                o = self.trader.query_order(oid)
                if not o or o.order_status not in TERMINAL_STATUS:
                    continue
                self.handled_terminal_orders.add(oid)
                traded = int(getattr(o, 'traded_volume', 0) or 0)
                if traded > 0:
                    continue
                msg = (getattr(o, 'status_msg', '') or '').strip()
                action = 'buy_underfunded' if _is_insufficient_funds(msg) else 'buy_reject_retry_later'
                delay = (self.executor.UNDERFUNDED_BACKOFF_SEC if _is_insufficient_funds(msg)
                         else self.executor.ORDER_REJECT_RETRY_SEC)
                self.retry_after[code] = now + delay
                self.record_action(action, code=code, order_id=oid,
                                   order_type=xtconstant.STOCK_BUY,
                                   shares=int(getattr(o, 'order_volume', 0) or 0),
                                   msg=msg)

    def _waiting_for_retry(self, now: float) -> bool:
        for code in self.buy_seq:
            if self._remaining(code) < min_buy_shares(code):
                continue
            if self.retry_after.get(code, 0.0) > now:
                return True
        return False

    # ── 主循环 ─────────────────────────────────────────────
    def _buy_loop(self):
        deadline = self._monitor_deadline()
        last_log = time.time()
        while time.time() < deadline:
            self._handle_terminal_orders()
            if time.time() - last_log >= 30:
                trading_logger.info(
                    f"[BuyMonitor] 挂单监控中: 已提交 {len(self.submitted)} 笔, "
                    f"卖单回款在途={self._sells_pending()}")
                last_log = time.time()
            progressed = self._submit_affordable_buys()
            if self._all_done_or_blocked():
                return
            if not progressed:
                now = time.time()
                if self._waiting_for_retry(now):
                    time.sleep(self.executor.MONITOR_POLL_SEC)
                    continue
                cash = self._available_cash()
                if not self._sells_pending() and not self._can_afford_any(cash):
                    trading_logger.warning(
                        f"[BuyMonitor] 可用资金 {cash:.2f} 不足以买入任何剩余标的,结束挂单提交")
                    return
                time.sleep(self.executor.MONITOR_POLL_SEC)

    def _submit_affordable_buys(self) -> bool:
        """按 TopN 顺序提交当前现金买得起的买单。返回本轮是否有新委托。"""
        progressed = False
        now = time.time()
        for code in self.buy_seq:
            if now < self.retry_after.get(code, 0.0):
                continue
            rem = self._remaining(code)
            min_lot = min_buy_shares(code)
            if rem < min_lot:
                continue
            cash = self._available_cash()
            afford = floor_buy_shares(code, int(cash / self._unit_cost(code)))
            shares = floor_buy_shares(code, min(rem, afford))
            if shares < min_lot:
                continue
            r = self._submit_buy(code, shares)
            if not r:
                self.retry_after[code] = time.time() + self.executor.ORDER_REJECT_RETRY_SEC
                self.record_action('buy_submit_failed_retry_later', code=code,
                                   order_type=xtconstant.STOCK_BUY, shares=shares)
                continue
            progressed = bool(r) or progressed
        return progressed

    def _submit_buy(self, code, shares):
        ml = min_buy_shares(code)
        shares = floor_buy_shares(code, shares)
        if shares < ml:
            return None
        open_px = self.prices.get(code, 0)
        limit_price = (self.executor._buy_protect_price(open_px, self.executor.BUY_PROTECT_PCT)
                       if open_px > 0 else None)
        r = self.executor._submit_buy_order(
            code, shares, self.signal_date, self.trade_date, limit_price=limit_price)
        if not r:
            return None
        r['submitted_at'] = time.time()
        self.orders_by_code[code].append(r['order_id'])
        self.submitted.append(r)
        self.record_action(
            'buy_submit', code=code, order_id=r['order_id'],
            order_type=xtconstant.STOCK_BUY, shares=shares,
            amount=shares * self._unit_cost(code),
            msg=f'limit={limit_price:.2f}' if limit_price else '',
        )
        return r


class RebalanceExecutor:
    """调仓下单执行器,依赖一个 `Trader` 实例进行实际委托。"""

    SLIPPAGE = LIVE_BUY_PRICE_BUFFER       # 缺涨停价时 unit_cost 回退缓冲(=开盘价×SLIPPAGE)
    BUY_FEE_RATE = LIVE_BUY_FEE_RATE       # 实盘券商冻结费率(佣金+过户费)
    BUY_PROTECT_PCT = 0.0     # 买入限价 = 开盘价（限价 ≤ 开盘价才能成交）
    SELL_PROTECT_PCT = 0.0    # 卖出限价 = 开盘价（限价 ≤ 开盘价才能成交）
    BUY_MONITOR_END = dt_time(14, 55)  # 资金不够时等卖出回款到收盘前,再挂后续买单
    BUY_MONITOR_DEADLINE_SEC = None    # 测试/闭市演练可覆盖为秒级
    UNDERFUNDED_BACKOFF_SEC = 3.0      # 资金不足废单后退避,等 QMT cash 更新
    ORDER_REJECT_RETRY_SEC = 60.0      # 价格范围/柜台废单后按原价低频重试
    SETTLE_WAIT_SEC = 5                # 提交结束后短暂沉淀回调;不撤未终态订单
    MONITOR_POLL_SEC = 0.3             # 买入资金门控轮询间隔
    # --skip 且真实时钟在闭市: order_stock 同步 API 可阻塞数分钟(非「废单慢」);
    # 此时跳过真实委托,仅收口战报/盘后。盘中委托仍走正常轮询超时。
    _TIMEOUT_FIELDS = (
        'BUY_MONITOR_DEADLINE_SEC', 'SETTLE_WAIT_SEC',
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
    def _buy_protect_price(open_price: float, protect_pct: float) -> float:
        """买入限价:开盘价×(1+protect_pct),按 A 股最小报价单位取整。"""
        tick = 0.001 if open_price < 1.0 else 0.01
        raw = open_price * (1 + protect_pct)
        return round(round(raw / tick) * tick, 3 if tick == 0.001 else 2)

    @staticmethod
    def _sell_protect_price(open_price: float, protect_pct: float) -> float:
        """卖出限价:开盘价×(1-protect_pct),按 A 股最小报价单位取整。"""
        tick = 0.001 if open_price < 1.0 else 0.01
        raw = open_price * (1 - protect_pct)
        return round(round(raw / tick) * tick, 3 if tick == 0.001 else 2)

    def _submit_sell_order(self, code, shares, signal_date, trade_date, *, limit_price=None):
        """提交卖出委托(限价单)。shares=-1 表示全部清仓。返回 {code, order_type, order_id, shares} 或 None。"""
        remark = f'rebalance signal={signal_date.isoformat()} trade={trade_date.isoformat()}'
        try:
            if shares < 0:
                pos = self.trader.query_stock_position(code)
                shares = int(getattr(pos, 'can_use_volume', 0) or 0) if pos else 0
                if shares < 100:
                    trading_logger.info(f"{code} 无可卖持仓，跳过清仓")
                    return None
            order_id = self.trader.order(
                xtconstant.STOCK_SELL, code, shares, limit_price, order_remark=remark)
            if order_id is None:
                trading_logger.info(f"{code} 无需卖出或委托未发出")
                return None
            px_msg = f' @{limit_price:.2f}' if limit_price else ''
            trading_logger.info(
                f"已提交卖出委托: {code} {shares}股{px_msg} order_id={order_id}")
            recorder.mark("提交卖出委托")
            return {'code': code, 'order_type': 'SELL', 'order_id': order_id, 'shares': shares}
        except ValueError as e:
            trading_logger.info(f"{code} 卖出前校验拦截: {e}")
        except Exception as e:
            trading_logger.warning(f"{code} 卖出委托失败: {e}")
        return None

    def _submit_buy_order(self, code, shares, signal_date, trade_date, *, limit_price=None):
        """提交买入委托(保护限价单)。返回 {code, order_type, order_id, shares} 或 None。"""
        remark = f'rebalance signal={signal_date.isoformat()} trade={trade_date.isoformat()}'
        try:
            order_id = self.trader.order(
                xtconstant.STOCK_BUY, code, shares, limit_price, order_remark=remark)
            px_msg = f' @{limit_price:.2f}' if limit_price else ''
            trading_logger.info(f"已提交买入委托: {code} * {shares} 股{px_msg} order_id={order_id}")
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
        for code, shares in sell_orders:
            trading_logger.info(
                f"[执行计划-卖] code={code} shares={shares} "
                f"open_price={prices.get(code, 0):.4f}")
        for code in buy_n_stocks:
            if code not in buy_allocations:
                continue
            trading_logger.info(
                f"[执行计划-买] code={code} shares={int(buy_allocations[code])} "
                f"open_price={prices.get(code, 0):.4f} "
                f"freeze_price={limit_prices.get(code, 0):.4f}")

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
            sell_monitor = SellMonitor(self, sell_orders, signal_date, trade_date, prices=prices)
        buy_monitor = None
        if execute_buy and buy_allocations:
            buy_monitor = BuyMonitor(
                self, buy_allocations, buy_n_stocks, prices,
                signal_date, trade_date, limit_prices=limit_prices)

        # 持久挂单：先把卖单全部报出,随后买单按 QMT 可用现金尽量报出。
        # 已报订单不撤,资金不够的买单等卖出回款到账后继续挂。
        if sell_monitor:
            sell_submitted = sell_monitor._submit_initial_sells()
        if buy_monitor:
            if sell_monitor:
                buy_monitor.sell_monitor = sell_monitor
            buy_submitted = buy_monitor.run()

        submitted = sell_submitted + buy_submitted
        self._wait_terminal(submitted)
        self._summarize(submitted)
        return submitted
