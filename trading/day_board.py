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

from trading.helper import get_order_status_label, get_order_type_label
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


def _fmt_money(v: float | None) -> str:
    if v is None:
        return '-'
    if abs(v) >= 1e4:
        return f"¥{v/1e4:.1f}w"
    return f"¥{v:,.0f}"


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
            self._plan: dict[str, dict] = {}        # code -> {direction, name, est_volume, est_price, est_amount, reason, plan_seq}
            self._orders: dict[int, dict] = {}      # order_id -> 订单最新状态
            self._trades: list[dict] = []            # 成交记录列表
            self._errors: list[dict] = []            # 订单错误
            self._equity: Optional[float] = None
            self._position_count: int = 0
            self._base_target: Optional[float] = None
            self._buy_n: Optional[int] = None
            self._dirty: bool = False
            self._finalized: bool = False
            if self._timer:
                try: self._timer.cancel()
                except Exception: pass
                self._timer = None

    # ─── 公开 API ─────────────────────────────────────
    def start_session(self, *, trade_date: date, plan_rows: list[dict],
                      equity: float | None = None, position_count: int = 0,
                      base_target: float | None = None, buy_n: int | None = None):
        """09:25 before_trade 调用：开启当日战报。"""
        with self._lock:
            self.reset_state()
            self._trade_date = trade_date
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
            try:
                from data.db import get_stock_detail
                detail = get_stock_detail(code)
                name = (detail.get('InstrumentName', '') if detail else '').strip()
            except Exception:
                pass
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

        return {
            'rows': rows,
            'buckets': bucket_count,
            'buy_done_amt': buy_done_amt,
            'sell_done_amt': sell_done_amt,
        }

    def _build_row(self, code: str, plan: dict | None, order_type: int,
                    orders: list[dict]) -> dict:
        """单只股票（同方向）的聚合行。"""
        name = (plan or {}).get('name', '') or code
        est_vol = (plan or {}).get('est_volume', 0)
        est_price = (plan or {}).get('est_price', 0.0)
        est_amount = (plan or {}).get('est_amount', 0.0)
        reason = (plan or {}).get('reason', '')

        if not orders:
            # 计划中但未下单（如计划被合法性过滤掉）
            limit = (plan or {}).get('limit_status', 'ok')
            if limit != 'ok':
                bucket = 'failed'
                label = '过滤'
                status_msg = reason or limit
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

    def _build_card(self) -> dict:
        agg = self._aggregate()
        rows = agg['rows']
        buckets = agg['buckets']

        # 排序：失败 → 部分 → 待成 → 已成；同 bucket 内按方向（卖出在前，与多退少补流程一致）
        bucket_order = {'failed': 0, 'partial': 1, 'pending': 2, 'done': 3}
        rows.sort(key=lambda r: (
            bucket_order.get(r['bucket'], 9),
            0 if r['order_type'] == xtconstant.STOCK_SELL else 1,
            r['name'],
        ))

        # header
        n_total = len(rows)
        n_done = buckets['done']
        n_partial = buckets['partial']
        n_failed = buckets['failed']
        n_pending = buckets['pending']

        if self._finalized:
            if n_failed > 0:
                tpl = LarkMsgLevel.Danger
                status_text = f'已完成 · {n_done}成 / {n_failed}失败'
            elif n_partial > 0:
                tpl = LarkMsgLevel.Warning
                status_text = f'已完成 · {n_done}成 / {n_partial}部分'
            else:
                tpl = LarkMsgLevel.Success
                status_text = f'已完成 · 全部成交 ({n_done}/{n_total})'
        else:
            if n_failed > 0:
                tpl = LarkMsgLevel.Danger
            elif n_partial > 0 or n_pending > 0:
                tpl = LarkMsgLevel.Info
            else:
                tpl = LarkMsgLevel.Success
            status_text = f'进行中 · ✅ {n_done} / ⚠ {n_partial} / ❌ {n_failed} / ⏳ {n_pending}'

        title = f'📋 调仓战报 @ {self._trade_date.isoformat()}'
        subtitle = status_text

        # 概览
        summary_parts = []
        if self._equity is not None:
            summary_parts.append(f"**权益** ¥{self._equity:,.0f}")
        if self._position_count:
            summary_parts.append(f"**持仓** {self._position_count} 只")
        if self._buy_n:
            summary_parts.append(f"**目标仓** {self._buy_n} 只")
        if self._base_target:
            summary_parts.append(f"**单仓** ¥{self._base_target:,.0f}")
        summary_parts.append(
            f"**成交** 买 {_fmt_money(agg['buy_done_amt'])} · 卖 {_fmt_money(agg['sell_done_amt'])}"
        )
        summary_md = '  ·  '.join(summary_parts)

        # 表格行
        table_rows = []
        for r in rows:
            traded_str = (
                f"{r['traded_price']:.2f} × {r['traded_volume']:,}"
                if r['traded_volume'] else '-'
            )
            table_rows.append({
                'name': r['name'],
                'dir': _direction_md(r['order_type']),
                'plan': f"{r['est_price']:.2f} × {r['est_volume']:,}" if r['est_volume'] else '-',
                'traded': traded_str,
                'amount': _fmt_money(r['traded_amount']) if r['traded_amount'] else (
                    '-' if not r['est_amount'] else f'<font color="grey">{_fmt_money(r["est_amount"])}</font>'),
                'status': _bucket_md(r['bucket'], r['status_label']),
                'msg': r['status_msg'] or '',
            })

        # 错误日志（仅 errors）
        elements = [md_div(summary_md)]
        if table_rows:
            elements.append({'tag': 'hr'})
            elements.append(md_div(
                f"**📋 订单进度** · 共 {n_total} 单 · 排序：失败 → 部分 → 待成 → 已成"))
            elements.append(make_v2_table(
                columns=[
                    {'name': 'name', 'display_name': '名称', 'horizontal_align': 'left'},
                    {'name': 'dir', 'display_name': '方向', 'horizontal_align': 'center'},
                    {'name': 'plan', 'display_name': '计划价×量', 'horizontal_align': 'right'},
                    {'name': 'traded', 'display_name': '成交价×量', 'horizontal_align': 'right'},
                    {'name': 'amount', 'display_name': '金额', 'horizontal_align': 'right'},
                    {'name': 'status', 'display_name': '状态', 'horizontal_align': 'left'},
                    {'name': 'msg', 'display_name': '备注', 'horizontal_align': 'left'},
                ],
                rows=table_rows,
                element_id='orders_progress',
                page_size=20,
            ))
        if self._errors:
            elements.append({'tag': 'hr'})
            err_md_lines = [f'**🚨 订单错误 {len(self._errors)} 条**']
            for e in self._errors[-10:]:  # 最近 10 条
                tm = e['at'].strftime('%H:%M:%S')
                code = e.get('code') or '-'
                name = e.get('name') or ''
                label = f"{code} {name}".strip()
                err_md_lines.append(f'<font color="red">[{tm}] {label}: {e["msg"]}</font>')
            elements.append(md_div('\n\n'.join(err_md_lines)))

        elements.append({'tag': 'hr'})
        footer = f'<font color="grey">更新时间 {datetime.now().strftime("%H:%M:%S")}'
        if self._finalized:
            footer += ' · 已锁定'
        footer += '</font>'
        elements.append(md_div(footer))

        return make_v2_card(title=title, level=tpl, subtitle=subtitle,
                            elements=elements)


# 全局单例
day_board = TradingDayBoard()
