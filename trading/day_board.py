"""调仓战报 — 实盘一天一张「可更新」飞书卡片。

核心动机：解决「订单/成交回调风暴」导致飞书消息刷屏的问题。
设计要点：
1. **单张聚合卡片**：盘前发出第一版（计划清单），订单/成交回调实时 update 同一张卡片
2. **节流 debounce**：连续回调短时间内合并为一次更新（默认 300ms）
3. **状态聚合分类**：⏳ 待成 / ✅ 已成 / ⚠️ 部分 / ❌ 废单 — 一目了然看到失败
4. **header 颜色随进度**：wathet（进行中）→ orange（部分失败）→ red（大量失败）→ green（全成）
5. **finalize 锁定**：盘后冻结，不再被新事件污染

线程安全：watcher 回调来自 QMT 线程，scheduler/main 来自主线程，
        都用 RLock 保护内部状态；卡片渲染只读取快照。
"""
from __future__ import annotations
import threading
from datetime import date, datetime
from typing import Optional

from xtquant import xtconstant

from data.db.stock_name import get_stock_name_at_date
from trading.helper import get_order_status_label, get_order_type_label
from trading.lark.format import (fmt_pct, fmt_diff_money, fmt_shares,
                                  diff_md, diff_shares_md)
from trading.lark.sender import lark_sender, LarkMsgLevel, make_v2_card, make_v2_table, md_div
from trading.logger import trading_logger


# 订单状态聚合（按对账价值排序：失败置顶 → 部分 → 待成 → 已成）
_STATUS_FAILED = {xtconstant.ORDER_CANCELED, xtconstant.ORDER_JUNK, xtconstant.ORDER_PART_CANCEL}
_STATUS_DONE = {xtconstant.ORDER_SUCCEEDED}
_STATUS_PARTIAL = {xtconstant.ORDER_PARTSUCC} if hasattr(xtconstant, 'ORDER_PARTSUCC') else set()


def _status_bucket(status: int) -> str:
    if status in _STATUS_FAILED:
        return 'failed'
    if status in _STATUS_PARTIAL:
        return 'partial'
    if status in _STATUS_DONE:
        return 'done'
    return 'pending'


def _bucket_emoji(bucket: str) -> str:
    return {'failed': '❌', 'partial': '⚠️', 'done': '✅', 'pending': '⏳'}[bucket]


def _bucket_md(bucket: str, label: str) -> str:
    if bucket == 'failed':
        return f'<font color="red">❌ {label}</font>'
    if bucket == 'partial':
        return f'<font color="orange">⚠️ {label}</font>'
    if bucket == 'done':
        return f'<font color="green">✅ {label}</font>'
    return f'<font color="grey">⏳ {label}</font>'


def _direction_md(order_type: int) -> str:
    if order_type == xtconstant.STOCK_BUY:
        return '<font color="red">买入</font>'
    if order_type == xtconstant.STOCK_SELL:
        return '<font color="green">卖出</font>'
    return get_order_type_label(order_type)



def extract_bt_reference(bt_result: dict) -> dict:
    """从 seed-replay bt_result 抽取 T 日「回测操作 + 回测持仓」参考，供战报实时对账。

    回测端继承 T-1 实盘状态（现金+持仓），结果固定不变；实盘端随成交回调累计逼近回测。
    返回 {'buy': {code: {shares, amount}}, 'sell': {...}, 'positions': {code: shares}}。
    """
    snaps = (bt_result or {}).get('daily_snapshots') or []
    t_snap = snaps[-1] if snaps else {}
    bt_buy = {}
    for d in t_snap.get('executed_buy_details', []) or []:
        bt_buy[d['code']] = {'shares': int(d['shares']),
                             'amount': float(d['shares']) * float(d['price'])}
    bt_sell = {}
    for d in t_snap.get('executed_sell_details', []) or []:
        bt_sell[d['code']] = {'shares': int(d['shares']),
                              'amount': float(d['shares']) * float(d['price'])}
    bt_pos = {p['code']: int(p['volume'])
              for p in t_snap.get('positions_eod', []) or []}
    return {'buy': bt_buy, 'sell': bt_sell, 'positions': bt_pos}


class TradingDayBoard:
    """全天聚合战报卡片。"""

    _DEBOUNCE_SEC = 0.3  # 节流时长

    def __init__(self):
        self._lock = threading.RLock()
        self._timer: Optional[threading.Timer] = None
        self.reset_state()

    # ─── 状态管理 ─────────────────────────────────────
    def reset_state(self):
        with self._lock:
            self._message_id: Optional[str] = None
            self._trade_date: Optional[date] = None
            self._plan: dict[str, dict] = {}
            self._orders: dict[int, dict] = {}
            self._trades: list[dict] = []
            self._errors: list[dict] = []
            self._equity: Optional[float] = None
            self._position_count: int = 0
            self._base_target: Optional[float] = None
            self._buy_n: Optional[int] = None
            self._bt_ref: Optional[dict] = None
            self._y_positions: dict[str, int] = {}
            # 实盘 T 日持仓权威快照（盘后从 positions_{T}.parquet 灌入，不走 QMT）。
            # None = 未灌入（盘中按 y_positions+成交实时重建）；非 None = 直接采用。
            self._live_positions: Optional[dict[str, int]] = None
            self._name_cache: dict[str, str] = {}
            self._bt_return: Optional[float] = None       # 回测预期日收益
            self._live_pnl: Optional[float] = None         # 实盘日盈亏（盘后填入）
            self._live_return: Optional[float] = None      # 实盘日收益率（盘后填入）
            self._dirty: bool = False
            self._finalized: bool = False
            if self._timer:
                try: self._timer.cancel()
                except Exception: pass
                self._timer = None

    # ─── 公开 API ─────────────────────────────────────
    def start_session(self, *, trade_date: date, plan_rows: list[dict],
                      equity: float | None = None, position_count: int = 0,
                      base_target: float | None = None, buy_n: int | None = None,
                      bt_ref: dict | None = None, bt_daily_return: float | None = None,
                      y_positions: dict | None = None):
        """09:25:10 before_trade 调用：开启当日战报。

        bt_ref / y_positions：盘前 seed-replay（继承 T-1 实盘状态）的回测参考 + T-1 实盘持仓，
        用于「回测 vs 实盘」实时对账（T 日操作 + T 日持仓）。缺失时退回纯订单进度战报。
        """
        with self._lock:
            self.reset_state()
            self._trade_date = trade_date
            self._bt_ref = bt_ref
            self._y_positions = dict(y_positions or {})
            for row in plan_rows:
                self._plan[row['code']] = {
                    'direction': row.get('direction', 'buy'),
                    'name': row.get('name', '') or '',
                    'est_volume': int(row.get('est_volume', 0) or 0),
                    'est_price': float(row.get('est_price', 0) or 0),
                    'est_amount': float(row.get('est_amount', 0) or 0),
                    'reason': row.get('reason', '') or '',
                    'plan_seq': int(row.get('plan_seq', 0) or 0),
                    'limit_status': row.get('limit_status', 'ok'),
                }
            self._equity = equity
            self._position_count = position_count
            self._base_target = base_target
            self._buy_n = buy_n
            self._bt_return = bt_daily_return
            self._live_pnl = None
            self._live_return = None
            self._preload_names()
        # 立即发卡（debounce 0 触发，第一次必须建立 message_id）
        self._push_now()

    def record_order(self, order):
        """on_stock_order 回调：更新订单最新状态。"""
        with self._lock:
            if self._finalized:
                return
            o = self._orders.get(order.order_id, {})
            o.update({
                'order_id': int(order.order_id),
                'code': order.stock_code,
                'order_type': int(order.order_type),
                'order_status': int(order.order_status),
                'order_volume': int(order.order_volume or 0),
                'traded_volume': int(getattr(order, 'traded_volume', 0) or 0),
                'price': float(order.price or 0),
                'traded_price': float(getattr(order, 'traded_price', 0) or 0),
                'status_msg': (order.status_msg or '').strip(),
                'updated_at': datetime.now(),
            })
            self._orders[order.order_id] = o
        self._mark_dirty()

    def record_trade(self, trade):
        """on_stock_trade 回调：追加一条成交记录。"""
        with self._lock:
            if self._finalized:
                return
            self._trades.append({
                'order_id': int(trade.order_id),
                'code': trade.stock_code,
                'order_type': int(trade.order_type),
                'price': float(trade.traded_price or 0),
                'volume': int(trade.traded_volume or 0),
                'amount': float(trade.traded_amount or 0),
                'at': datetime.now(),
            })
        self._mark_dirty()

    def record_order_error(self, err):
        """订单错误回调（已不发独立卡，全部聚合到战报）。"""
        code = getattr(err, 'stock_code', '') or ''
        with self._lock:
            if self._finalized:
                return
            name = self._plan.get(code, {}).get('name', '') if code else ''
        if not name and code:
            name = get_stock_name_at_date(code, self._trade_date) or ''
        with self._lock:
            if self._finalized:
                return
            self._errors.append({
                'code': code,
                'name': name,
                'msg': getattr(err, 'error_msg', '') or '',
                'at': datetime.now(),
            })
        self._mark_dirty()

    def feed_close_data(self, *, live_pnl: float | None = None,
                         live_return: float | None = None):
        """盘后填入实盘日收益数据，立即触发卡片更新。"""
        with self._lock:
            self._live_pnl = live_pnl
            self._live_return = live_return
        self._mark_dirty()

    def feed_bt_reference(self, bt_ref: dict | None, *,
                          bt_daily_return: float | None = None,
                          y_positions: dict | None = None):
        """补充/刷新「回测对账参考」并立即更新卡片。

        盘前 seed-replay 缺 T-1 种子（sim / 首日）时 bt_ref 为空，导致「目标 vs 实盘」
        三列全为「-」。盘后回测（连续回测）拿到真实目标后回灌进来，让卡片目标列有值。
        finalized 后不再接受刷新（盘后 finalize 会做最后一次 push）。
        """
        with self._lock:
            if self._finalized:
                return
            if bt_ref is not None:
                self._bt_ref = bt_ref
                self._preload_names()
            if bt_daily_return is not None:
                self._bt_return = bt_daily_return
            if y_positions is not None:
                self._y_positions = dict(y_positions)
        self._mark_dirty()

    def feed_live_fills(self, fills_df):
        """从 fills_{T}.parquet 灌入实盘成交（盘后对账用，**不走 QMT**）。

        动机：sim/replay/进程重启时没有 QMT 成交回调，self._trades 为空，
        导致「操作(实盘)」「实盘日志」全空。盘后 fills 已落地 parquet（实盘回调 +
        QMT 兜底回填都汇入它），直接读它即可，避免非交易时段查 QMT 卡死。
        权威替换 self._trades（parquet 已去重，比回调累计更完整）。
        """
        if fills_df is None or getattr(fills_df, 'empty', True):
            return
        trades = []
        for _, r in fills_df.iterrows():
            direction = str(r.get('direction', '') or '').strip()
            ot = xtconstant.STOCK_BUY if direction == 'buy' else xtconstant.STOCK_SELL
            trades.append({
                'order_id': int(r.get('order_id', 0) or 0),
                'code': r['code'],
                'order_type': ot,
                'price': float(r.get('price', 0) or 0),
                'volume': int(r.get('shares', 0) or 0),
                'amount': float(r.get('amount', 0) or 0),
                'at': datetime.now(),
            })
        with self._lock:
            if self._finalized:
                return
            self._trades = trades
        self._mark_dirty()

    def feed_live_positions(self, positions_df):
        """从 positions_{T}.parquet 灌入实盘 T 日持仓快照（盘后对账用，**不走 QMT**）。

        盘后用真实 EOD 持仓作「实盘持仓」列，使其与「目标持仓」可直接比对
        （理论上应差不多）。缺 T-1 快照时也不影响——直接采用 T 日权威快照，
        不再依赖 y_positions+成交重建。
        """
        if positions_df is None or getattr(positions_df, 'empty', True):
            return
        pos = {}
        for _, r in positions_df.iterrows():
            vol = int(r.get('volume', 0) or 0)
            if vol > 0:
                pos[r['code']] = vol
        with self._lock:
            if self._finalized:
                return
            self._live_positions = pos
            self._preload_names()
        self._mark_dirty()

    def finalize(self):
        """post_close 调用：立即刷新最终状态并锁定。"""
        with self._lock:
            if self._finalized:
                return
            self._finalized = True
            if self._timer:
                try: self._timer.cancel()
                except Exception: pass
                self._timer = None
        self._push_now()

    def _preload_names(self):
        """对比表/订单表：盘前一次性从本地 parquet 预热简称缓存（无网络）。"""
        codes: set[str] = set(self._plan) | set(self._y_positions)
        if self._live_positions:
            codes |= set(self._live_positions)
        if self._bt_ref:
            codes |= set(self._bt_ref.get('buy', {}))
            codes |= set(self._bt_ref.get('sell', {}))
            codes |= set(self._bt_ref.get('positions', {}))
        for code in codes:
            if code not in self._name_cache:
                self._name_cache[code] = get_stock_name_at_date(code, self._trade_date) or ''

    def _plan_visible(self, code: str, plan: dict) -> bool:
        """订单进度：仅展示有实盘下单意图的项；跳过的 topN 买入若回测也未交易则隐藏。"""
        if plan.get('direction') == 'sell':
            return int(plan.get('est_volume', 0) or 0) > 0
        if plan.get('limit_status', 'ok') == 'ok':
            return True
        if not self._bt_ref:
            return False
        bt = self._bt_ref
        return code in bt.get('buy', {}) or code in bt.get('sell', {})

    # ─── 内部 ─────────────────────────────────────
    def _mark_dirty(self):
        with self._lock:
            self._dirty = True
            if self._timer:
                try: self._timer.cancel()
                except Exception: pass
            self._timer = threading.Timer(self._DEBOUNCE_SEC, self._push_now)
            self._timer.daemon = True
            self._timer.start()

    def _push_now(self):
        """同步发出/更新战报卡片。"""
        with self._lock:
            self._dirty = False
            if not self._trade_date:
                return
            card = self._build_card()
            mid = self._message_id
            finalized = self._finalized
        try:
            if mid is None:
                new_id = lark_sender.send_card(card)
                if new_id:
                    with self._lock:
                        self._message_id = new_id
                    trading_logger.info(f"[Board] 战报已发送 message_id={new_id}")
                else:
                    trading_logger.warning("[Board] 战报首次发送未拿到 message_id")
            else:
                ok = lark_sender.update_card(mid, card)
                if not ok:
                    trading_logger.warning(f"[Board] 战报更新失败 message_id={mid}")
                elif finalized:
                    trading_logger.info(f"[Board] 战报已 finalize message_id={mid}")
        except Exception as e:
            trading_logger.warning(f"[Board] 战报推送异常: {e}")

    # ─── 卡片渲染 ─────────────────────────────────────
    def _aggregate(self) -> dict:
        """聚合订单状态：以 (code, direction) 为单位，关联 plan 和 orders。"""
        # plan 状态：未提交订单 → '待提交'；有订单 → 取最新状态
        # 按 code 聚合所有 order_id（一只股票可能有多笔订单：多退少补、补单等）
        per_code_orders: dict[tuple[str, int], list[dict]] = {}
        for o in self._orders.values():
            key = (o['code'], o['order_type'])
            per_code_orders.setdefault(key, []).append(o)

        # 合并 plan + orders 行
        rows = []
        plan_seen = set()
        for code, plan in self._plan.items():
            if not self._plan_visible(code, plan):
                continue
            direction = plan['direction']
            order_type = xtconstant.STOCK_BUY if direction == 'buy' else xtconstant.STOCK_SELL
            key = (code, order_type)
            plan_seen.add(key)
            orders_here = per_code_orders.get(key, [])
            rows.append(self._build_row(code, plan, order_type, orders_here))

        # 计划之外的订单（异常情况）
        for key, orders in per_code_orders.items():
            if key in plan_seen:
                continue
            code, order_type = key
            rows.append(self._build_row(code, None, order_type, orders))

        # 汇总
        bucket_count = {'failed': 0, 'partial': 0, 'done': 0, 'pending': 0}
        buy_done_amt = 0.0
        sell_done_amt = 0.0
        for r in rows:
            bucket_count[r['bucket']] += 1
            if r['traded_amount']:
                if r['order_type'] == xtconstant.STOCK_BUY:
                    buy_done_amt += r['traded_amount']
                else:
                    sell_done_amt += r['traded_amount']

        # 目标(计划)金额:plan 中各方向 est_amount 之和(买=少补目标,卖=换出目标)
        plan_buy_amt = sum(p['est_amount'] for p in self._plan.values()
                           if p['direction'] == 'buy')
        plan_sell_amt = sum(p['est_amount'] for p in self._plan.values()
                            if p['direction'] == 'sell')

        return {
            'rows': rows,
            'buckets': bucket_count,
            'buy_done_amt': buy_done_amt,
            'sell_done_amt': sell_done_amt,
            'plan_buy_amt': plan_buy_amt,
            'plan_sell_amt': plan_sell_amt,
        }

    def _build_row(self, code: str, plan: dict | None, order_type: int,
                    orders: list[dict]) -> dict:
        """单只股票（同方向）的聚合行。"""
        name = self._name_of(code)
        est_vol = (plan or {}).get('est_volume', 0)
        est_price = (plan or {}).get('est_price', 0.0)
        est_amount = (plan or {}).get('est_amount', 0.0)
        reason = (plan or {}).get('reason', '')

        if not orders:
            # 计划中但未下单：直接用 plan 里的具体原因
            limit = (plan or {}).get('limit_status', 'ok')
            if limit != 'ok':
                # 仅「真实闸门」(缺开盘价/合法性/资金不足) 算失败标红；
                # 「已达标/未触发少补/无实盘基线」是正常未下单，按待处理(灰)展示。
                _block = any(k in reason for k in ('合法性', '缺开盘价', '资金', '废'))
                bucket = 'failed' if _block else 'pending'
                label = reason or '未下单'
                status_msg = ''
            else:
                bucket = 'pending'
                label = '待下单'
                status_msg = ''
            return {
                'code': code, 'name': name, 'order_type': order_type,
                'est_volume': est_vol, 'est_price': est_price, 'est_amount': est_amount,
                'traded_volume': 0, 'traded_price': 0.0, 'traded_amount': 0.0,
                'bucket': bucket, 'status_label': label,
                'status_msg': status_msg,
            }

        # 取最新一笔订单（按 updated_at），多笔订单（拆单）合并量
        orders_sorted = sorted(orders, key=lambda o: o.get('updated_at') or datetime.min)
        latest = orders_sorted[-1]
        bucket = _status_bucket(latest['order_status'])
        label = latest.get('status_label') or get_order_status_label(latest['order_status'])

        # 累计成交（所有订单）
        traded_vol_total = sum(o.get('traded_volume', 0) for o in orders)
        traded_amt_total = 0.0
        # 用 fills 累加金额更准；缺则按 traded_price × traded_volume
        fills_for_code = [t for t in self._trades if t['code'] == code and t['order_type'] == order_type]
        if fills_for_code:
            traded_amt_total = sum(t['amount'] for t in fills_for_code)
            traded_vol_total = max(traded_vol_total, sum(t['volume'] for t in fills_for_code))
            avg_price = traded_amt_total / traded_vol_total if traded_vol_total else 0
        else:
            avg_price = latest.get('traded_price', 0)
            traded_amt_total = avg_price * traded_vol_total

        return {
            'code': code, 'name': name, 'order_type': order_type,
            'est_volume': est_vol, 'est_price': est_price, 'est_amount': est_amount,
            'traded_volume': traded_vol_total,
            'traded_price': avg_price,
            'traded_amount': traded_amt_total,
            'bucket': bucket, 'status_label': label,
            'status_msg': latest.get('status_msg', ''),
        }

    def _name_of(self, code: str) -> str:
        """对比表/订单表：plan 缓存 → get_stock_name_at_date；无简称则显示 code。"""
        if code in self._name_cache:
            cached = self._name_cache[code]
            if cached:
                return cached
        p = self._plan.get(code)
        if p and p.get('name'):
            self._name_cache[code] = p['name']
            return p['name']
        nm = get_stock_name_at_date(code, self._trade_date) or ''
        self._name_cache[code] = nm
        return nm or code

    def _compute_comparison(self) -> dict | None:
        """回测 vs 实盘：T 日操作（净买卖额）+ T 日持仓（股数）对比。

        - 回测端：盘前 seed-replay（继承 T-1 实盘现金+持仓），固定不变。
        - 实盘端：实时成交累计（self._trades），随订单进度逼近回测；
          实盘 T 日持仓 = T-1 持仓 + 今日买入 − 今日卖出。
        无 bt_ref（首日/缺 T-1 快照）时返回 None。
        """
        if not self._bt_ref:
            return None
        bt_buy = self._bt_ref.get('buy', {})
        bt_sell = self._bt_ref.get('sell', {})
        bt_pos = self._bt_ref.get('positions', {})

        # 实盘成交累计（按 code）
        live_buy_amt: dict[str, float] = {}
        live_buy_sh: dict[str, int] = {}
        live_sell_amt: dict[str, float] = {}
        live_sell_sh: dict[str, int] = {}
        for t in self._trades:
            code = t['code']
            if t['order_type'] == xtconstant.STOCK_BUY:
                live_buy_amt[code] = live_buy_amt.get(code, 0.0) + t['amount']
                live_buy_sh[code] = live_buy_sh.get(code, 0) + t['volume']
            else:
                live_sell_amt[code] = live_sell_amt.get(code, 0.0) + t['amount']
                live_sell_sh[code] = live_sell_sh.get(code, 0) + t['volume']

        # ── T 日操作对比（净买入额/手数 = 买 − 卖）──
        op_codes = set(bt_buy) | set(bt_sell) | set(live_buy_amt) | set(live_sell_amt)
        op_rows = []
        for code in op_codes:
            bt_net = (bt_buy.get(code, {}).get('amount', 0.0)
                      - bt_sell.get(code, {}).get('amount', 0.0))
            live_net = live_buy_amt.get(code, 0.0) - live_sell_amt.get(code, 0.0)
            bt_net_sh = bt_buy.get(code, {}).get('shares', 0) - bt_sell.get(code, {}).get('shares', 0)
            live_net_sh = live_buy_sh.get(code, 0) - live_sell_sh.get(code, 0)
            if bt_net == 0 and live_net == 0 and bt_net_sh == 0 and live_net_sh == 0:
                continue
            op_rows.append({'code': code,
                            'bt_net': bt_net, 'live_net': live_net, 'diff': live_net - bt_net,
                            'bt_net_sh': bt_net_sh, 'live_net_sh': live_net_sh,
                            'sh_diff': live_net_sh - bt_net_sh})
        op_rows.sort(key=lambda r: abs(r['diff']), reverse=True)
        op_totals = {'bt': sum(r['bt_net'] for r in op_rows),
                     'live': sum(r['live_net'] for r in op_rows)}

        # ── T 日持仓对比（股数）──
        # 实盘持仓优先用盘后灌入的 positions_{T} 权威快照（_live_positions）；
        # 盘中未灌入时按 T-1 持仓 + 今日成交实时重建。
        pos_codes = set(bt_pos) | set(self._y_positions) | set(live_buy_sh) | set(live_sell_sh)
        if self._live_positions is not None:
            pos_codes |= set(self._live_positions)
        pos_rows = []
        for code in pos_codes:
            if self._live_positions is not None:
                live_sh = self._live_positions.get(code, 0)
            else:
                live_sh = (self._y_positions.get(code, 0)
                           + live_buy_sh.get(code, 0) - live_sell_sh.get(code, 0))
            bt_sh = bt_pos.get(code, 0)
            if live_sh == 0 and bt_sh == 0:
                continue
            pos_rows.append({'code': code, 'bt_shares': bt_sh, 'live_shares': live_sh,
                             'diff': live_sh - bt_sh})
        pos_rows.sort(key=lambda r: (r['diff'] == 0, -abs(r['diff'])))
        pos_totals = {'bt': sum(r['bt_shares'] for r in pos_rows),
                      'live': sum(r['live_shares'] for r in pos_rows),
                      'aligned': sum(1 for r in pos_rows if r['diff'] == 0),
                      'total': len(pos_rows)}

        return {'op_rows': op_rows, 'op_totals': op_totals,
                'pos_rows': pos_rows, 'pos_totals': pos_totals}

    # ─── 单表卡片渲染：5 列统一表 ──────────────────────

    @staticmethod
    def _sh_diff_text(bt_sh: int, live_sh: int) -> str:
        """diff差异 列：手数差的人类可读描述。"""
        d = live_sh - bt_sh
        if d == 0:
            return '<font color="grey">—</font>'
        if bt_sh >= 0:  # 目标净买（或不动）
            word = '多买' if d > 0 else '少买'
        else:           # 目标净卖
            word = '少卖' if d > 0 else '多卖'
        color = 'red' if d > 0 else 'green'
        return f'<font color="{color}">实盘{word}{abs(d)}股</font>'

    @staticmethod
    def _pos_diff_text(bt_sh: int, live_sh: int) -> str:
        """持仓差异 列：实盘相对「目标持仓」的偏离（权威，与持仓列口径一致）。

        以 T 日终持仓（positions_{T} vs 回测 positions_eod）为唯一口径，
        避免「操作」列受 fills 不完整 / 缺 T-1 链影响而与持仓列自相矛盾。
        """
        d = live_sh - bt_sh
        if d == 0:
            return '<font color="grey">对齐</font>'
        color = 'red' if d > 0 else 'orange'
        word = '多持' if d > 0 else '少持'
        return f'<font color="{color}">实盘{word}{abs(d)}股</font>'

    def _build_card(self) -> dict:
        agg = self._aggregate()
        cmp = self._compute_comparison()

        # ── 合并数据源 → 每只股票一行 ──
        pos_map = {}  # code → {bt_shares, live_shares, diff}
        ops_map = {}  # code → {bt_net, live_net, bt_net_sh, live_net_sh, sh_diff}
        if cmp:
            for r in cmp['pos_rows']:
                pos_map[r['code']] = r
            for r in cmp['op_rows']:
                ops_map[r['code']] = r
        log_map: dict[str, list] = {}
        for r in agg['rows']:
            log_map.setdefault(r['code'], []).append(r)

        all_codes = set(pos_map) | set(ops_map) | set(log_map) | set(self._plan) | set(self._y_positions)
        if self._live_positions:
            all_codes |= set(self._live_positions)

        table = []
        for code in sorted(all_codes):
            p = pos_map.get(code, {})
            o = ops_map.get(code, {})
            logs = log_map.get(code, [])

            name = self._name_of(code)

            # 列1: 持仓手数 — 目标 vs 实盘
            bt_sh = p.get('bt_shares', 0)
            live_sh = p.get('live_shares', 0)
            if bt_sh or live_sh:
                if bt_sh == live_sh:
                    pos_cell = f'<font color="grey">目标{fmt_shares(live_sh)} vs 实盘{fmt_shares(live_sh)}</font>'
                else:
                    pos_cell = f'目标{fmt_shares(bt_sh)} vs 实盘{fmt_shares(live_sh)}'
            else:
                pos_cell = '-'

            # 列2: 操作 — 目标净买卖 vs 实盘净买卖（手数）
            bt_net_sh = o.get('bt_net_sh', 0)
            live_net_sh = o.get('live_net_sh', 0)
            if bt_net_sh or live_net_sh:
                ops_cell = f'目标{bt_net_sh:+d} vs 实盘{live_net_sh:+d}'
            else:
                ops_cell = '-'

            # 列3: 持仓差异 — 以「目标持仓 vs 实盘持仓」为权威口径（与列1一致），
            # 不再用操作差（fills 不完整时会与持仓列矛盾）。
            diff_cell = self._pos_diff_text(bt_sh, live_sh) if (bt_sh or live_sh) else '-'

            # 列4: 实盘日志
            log_parts = []
            if logs:
                for l in logs:
                    if l['bucket'] == 'pending' and bt_sh == live_sh and bt_sh > 0:
                        s = '<font color="green">✅ 已达标</font>'
                    else:
                        s = _bucket_md(l['bucket'], l['status_label'])
                        if l.get('status_msg'):
                            s += f' <font color="red">{l["status_msg"]}</font>'
                    log_parts.append(s)
            elif code in self._plan:
                plan = self._plan[code]
                if plan.get('limit_status') != 'ok':
                    log_parts.append(f'<font color="red">{plan.get("reason","未下单")}</font>')
                elif bt_sh == live_sh and bt_sh > 0:
                    log_parts.append('<font color="green">✅ 已达标</font>')
                else:
                    log_parts.append('<font color="grey">⏳ 待下单</font>')
            else:
                log_parts.append('-')
            log_cell = '\n'.join(log_parts) if log_parts else '-'

            table.append({'name': name, 'pos': pos_cell, 'ops': ops_cell,
                          'diff': diff_cell, 'log': log_cell})

        # 合计行
        if cmp:
            pt = cmp['pos_totals']
            ot = cmp['op_totals']
            total_bt_sh = sum(r.get('bt_net_sh', 0) for r in ops_map.values())
            total_live_sh = sum(r.get('live_net_sh', 0) for r in ops_map.values())
            table.append({
                'name': '— 合计 —',
                'pos': f'目标{fmt_shares(pt["bt"])} vs 实盘{fmt_shares(pt["live"])}',
                'ops': f'目标{total_bt_sh:+d} vs 实盘{total_live_sh:+d}',
                'diff': self._pos_diff_text(pt["bt"], pt["live"]),
                'log': f'<font color="grey">对齐 {pt["aligned"]}/{pt["total"]} 只</font>',
            })

        title_md = '**T日操作对比 / T日持仓对比**' if cmp else '**目标→成交**'
        elements = [md_div(title_md), make_v2_table(
            columns=[
                {'name': 'name', 'display_name': '股票', 'horizontal_align': 'left'},
                {'name': 'pos', 'display_name': '持仓手数(目标vs实盘)', 'horizontal_align': 'right'},
                {'name': 'ops', 'display_name': '操作(目标vs实盘)', 'horizontal_align': 'right'},
                {'name': 'diff', 'display_name': '持仓差异', 'horizontal_align': 'right'},
                {'name': 'log', 'display_name': '实盘日志', 'horizontal_align': 'left'},
            ],
            rows=table, element_id='daily_tbl', page_size=20)]

        # footer
        elements.append({'tag': 'hr'})
        footer = f'<font color="grey">更新 {datetime.now().strftime("%H:%M:%S")}'
        if self._finalized:
            footer += ' · 已锁定'
        footer += '</font>'
        elements.append(md_div(footer))

        # header: 日收益在副标题（盘前显示回测目标，盘后更新实盘）
        parts = []
        if self._live_pnl is not None:
            parts.append(f"实盘盈亏 {fmt_diff_money(self._live_pnl)}")
            if self._live_return is not None:
                parts.append(f"实盘 {fmt_pct(self._live_return, sign=True)}")
        if self._bt_return is not None:
            parts.append(f"回测 {fmt_pct(self._bt_return, sign=True)}")
        if not parts:
            parts.append('交易中...')

        if self._finalized:
            live_ret = self._live_return
            tpl = LarkMsgLevel.Success if (live_ret or 0) >= 0 else LarkMsgLevel.Danger
        else:
            tpl = LarkMsgLevel.Info
        sub = ' · '.join(parts)

        return make_v2_card(
            title=f"实盘日报 @ {self._trade_date.isoformat()}",
            level=tpl, subtitle=sub, elements=elements)


# 全局单例
day_board = TradingDayBoard()
