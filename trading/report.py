"""AL-5 盘后 5 维 Diff 报告 — 实盘 vs 回测全链路对账。

5 个维度：
 1. 候选股（选股层）         实盘 plan all-buy   vs 回测 raw_buy_n_list
 2. 可交易（合法性层）       实盘 plan ok-buy    vs 回测 buy_n_list (tradable)
 3. 订单（执行层）           实盘 fills           vs 回测 executed_buy/sell_details
 4. 滑点（市场冲击层）       实盘 traded vs est  vs 回测 0
 5. 日 P&L（结果层）         实盘 daily_pnl      vs 回测 daily_pnl (重建)

数据来源全部是 parquet + 回测 dict，不依赖网络。
"""
from __future__ import annotations
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from data.db.stock_name import get_stock_name_at_date
from trading.lark.sender import lark_sender
from trading.logger import trading_logger

_TRADE_DIR = Path(__file__).resolve().parents[1] / "data" / "live_trades"


# ============================================================
# 格式化工具
# ============================================================

def _name(code: str, signal_date: date) -> str:
    return get_stock_name_at_date(code, signal_date) or ''


def _fmt_money(v: float | None) -> str:
    if v is None or pd.isna(v):
        return '-'
    if abs(v) >= 1e8:
        return f"¥{v/1e8:.2f}亿"
    if abs(v) >= 1e4:
        return f"¥{v/1e4:.1f}w"
    return f"¥{v:,.0f}"


def _fmt_pct(v: float | None, sign: bool = False) -> str:
    if v is None or pd.isna(v):
        return '-'
    return f"{v:+.2f}%" if sign else f"{v:.2f}%"


def _fmt_diff_money(v: float | None) -> str:
    if v is None or pd.isna(v):
        return '-'
    if abs(v) >= 1e4:
        return f"{v/1e4:+.1f}w"
    return f"{v:+,.0f}"


def _safe_div(a, b):
    return a / b if b else None


# ============================================================
# 回测端逐股 P&L 重建（与实盘 snapshot_positions 公式同口径）
# ============================================================

def _rebuild_backtest_per_stock_pnl(bt_result: dict) -> dict[str, dict]:
    """从（可能是多日的）回测结果重建 T 日 per-stock daily_pnl。

    公式与实盘 snapshot_positions 同口径：
        daily_pnl = (T_lp × T_vol) - (Y_lp × Y_vol) + S_amt_T - B_amt_T - fee_T
    其中：
        - T_lp × T_vol = T 日收盘市值（来自 daily_snapshots[-1].positions_eod）
        - Y_lp × Y_vol = T-1 日收盘市值（来自 daily_snapshots[-2].positions_eod，
                          多日回测时；单日回测时 = 0）
        - S/B/fee 仅取 trade_log 中 signal_date == T 的子集

    Returns:
        {code: {'volume', 'mv', 'avg_price', 'current_price',
                'buy_amount', 'sell_amount', 'fee', 'daily_pnl', 'daily_return_pct'}}
    """
    result: dict[str, dict] = {}
    snaps = bt_result.get('daily_snapshots') or []
    if not snaps:
        snaps = [{'positions_eod': bt_result.get('positions', []),
                  'signal_date': None, 'date': None}]

    # T 日 = 最后一个 snapshot；T-1 日 = 倒数第二个（如果存在）
    t_snap = snaps[-1]
    y_snap = snaps[-2] if len(snaps) >= 2 else None
    t_signal_str = t_snap.get('signal_date') or t_snap.get('date')

    # 兼容老接口：positions_eod 缺失时回退到 bt_result['positions']（仅 T 日）
    t_positions = t_snap.get('positions_eod')
    if t_positions is None:
        t_positions = bt_result.get('positions') or []
    t_pos_map = {p['code']: p for p in t_positions}
    y_pos_map = {p['code']: p for p in ((y_snap.get('positions_eod') if y_snap else None) or [])}

    # 仅累计 T 日（signal_date == T）的 trade，模拟"今日发生的资金流"
    trade_agg: dict[str, dict] = {}
    for tr in bt_result.get('trade_log', []):
        code = tr.get('code')
        if not code:
            continue
        # signal_date 在 account.buy_stock 时被设置；可能是 date 或 str
        sd = tr.get('signal_date')
        sd_str = sd.isoformat() if hasattr(sd, 'isoformat') else str(sd) if sd else None
        if t_signal_str and sd_str and sd_str != t_signal_str:
            continue
        slot = trade_agg.setdefault(code, {'buy_amt': 0.0, 'sell_amt': 0.0,
                                          'buy_vol': 0, 'sell_vol': 0, 'fee': 0.0})
        if tr['action'] == 'buy':
            slot['buy_amt'] += float(tr.get('amount', 0))
            slot['buy_vol'] += int(tr.get('volume', 0))
        else:
            slot['sell_amt'] += float(tr.get('amount', 0))
            slot['sell_vol'] += int(tr.get('volume', 0))
        slot['fee'] += float(tr.get('commission', 0))

    all_codes = set(trade_agg.keys()) | set(t_pos_map.keys()) | set(y_pos_map.keys())
    for code in all_codes:
        p = t_pos_map.get(code)
        yp = y_pos_map.get(code)
        agg = trade_agg.get(code, {'buy_amt': 0.0, 'sell_amt': 0.0,
                                    'buy_vol': 0, 'sell_vol': 0, 'fee': 0.0})
        vol = int(p['volume']) if p else 0
        cp = float(p['current_price']) if p else 0.0
        mv = float(p['current_value']) if p else 0.0
        ap = float(p['avg_price']) if p else 0.0
        y_mv = float(yp['current_value']) if yp else 0.0

        # 多日同口径：daily_pnl = T_mv - Y_mv + S_amt - B_amt - fee
        daily_pnl = mv - y_mv + agg['sell_amt'] - agg['buy_amt'] - agg['fee']
        if y_mv > 0:
            daily_ret = daily_pnl / y_mv * 100
        elif agg['buy_amt'] > 0:
            daily_ret = daily_pnl / agg['buy_amt'] * 100
        else:
            daily_ret = None

        result[code] = {
            'volume': vol, 'mv': mv,
            'avg_price': ap, 'current_price': cp,
            'buy_amount': agg['buy_amt'], 'sell_amount': agg['sell_amt'],
            'fee': agg['fee'],
            'daily_pnl': daily_pnl,
            'daily_return_pct': daily_ret,
        }
    return result


# ============================================================
# PostCloseReport — 5 维 Diff
# ============================================================

class PostCloseReport:
    def __init__(self, trade_date: date):
        self.trade_date = trade_date
        # 5 维数据底座
        self._plan: pd.DataFrame | None = None       # plan_{T}.parquet
        self._fills: pd.DataFrame | None = None      # fills_{T}.parquet
        self._positions: pd.DataFrame | None = None  # positions_{T}.parquet
        self._bt: dict | None = None                 # 单日回测 result
        # 资产
        self._asset: float | None = None
        self._prev_asset: float | None = None
        self._net_cash_flow: float = 0.0
        # code → name 映射（feed 数据时累积，飞书卡片只显示名称）
        self._code_to_name: dict[str, str] = {}

    def _harvest_names(self, df: pd.DataFrame):
        if df is None or df.empty or 'code' not in df.columns or 'name' not in df.columns:
            return
        for _, r in df.iterrows():
            n = r.get('name')
            if n and isinstance(n, str) and n.strip():
                self._code_to_name[r['code']] = n.strip()

    def _name_of(self, code: str) -> str:
        """优先用已 feed 数据中的 name；否则查 db.stock_name；最后兜底用 code。"""
        n = self._code_to_name.get(code)
        if n:
            return n
        n = _name(code, self.trade_date)
        if n:
            self._code_to_name[code] = n
            return n
        return code  # 兜底

    def _names_of(self, codes: list[str]) -> list[str]:
        return [self._name_of(c) for c in codes]

    # ── 数据注入（全部从 parquet/dict 读，调用方负责拼装）─────────

    def feed_plan_df(self, df: pd.DataFrame):
        self._plan = df
        self._harvest_names(df)

    def feed_fills_df(self, df: pd.DataFrame):
        self._fills = df
        self._harvest_names(df)

    def feed_positions_df(self, df: pd.DataFrame):
        self._positions = df
        self._harvest_names(df)

    def feed_backtest(self, bt_result: dict):
        self._bt = bt_result

    def feed_asset(self, total_asset: Optional[float],
                   prev_asset: Optional[float], net_cash_flow: float = 0.0):
        self._asset = total_asset
        self._prev_asset = prev_asset
        self._net_cash_flow = net_cash_flow

    # ── 5 维构建 ────────────────────────────────────────────

    def _bt_snap(self) -> dict:
        """T 日 snapshot：多日连续回测时取最后一天。"""
        if not self._bt:
            return {}
        snaps = self._bt.get('daily_snapshots') or []
        return snaps[-1] if snaps else {}

    def _live_buy_candidates(self) -> list[str]:
        """实盘候选股 = plan 中所有 direction=buy 行（按 plan_seq 排序）。"""
        if self._plan is None or self._plan.empty:
            return []
        buys = self._plan[self._plan['direction'] == 'buy'].sort_values('plan_seq')
        return buys['code'].tolist()

    def _live_buy_tradable(self) -> list[str]:
        """实盘合法可交易 = plan 中 direction=buy & limit_status=ok。"""
        if self._plan is None or self._plan.empty:
            return []
        ok = self._plan[(self._plan['direction'] == 'buy')
                        & (self._plan['limit_status'] == 'ok')].sort_values('plan_seq')
        return ok['code'].tolist()

    def _live_buy_executed(self) -> dict[str, dict]:
        """实盘实际成交 = fills 中 direction=buy 聚合。返回 {code: {shares, amount, fee, price}}"""
        if self._fills is None or self._fills.empty:
            return {}
        buys = self._fills[self._fills['direction'] == 'buy']
        out = {}
        for code, grp in buys.groupby('code'):
            amt = float(grp['amount'].sum())
            sh = int(grp['shares'].sum())
            out[code] = {
                'shares': sh, 'amount': amt,
                'fee': float(grp['fee_est'].sum()),
                'price': amt / sh if sh > 0 else 0.0,
                'est_price': float(grp['est_price'].mean()) if 'est_price' in grp.columns and grp['est_price'].notna().any() else None,
            }
        return out

    def _live_sell_executed(self) -> dict[str, dict]:
        if self._fills is None or self._fills.empty:
            return {}
        sells = self._fills[self._fills['direction'] == 'sell']
        out = {}
        for code, grp in sells.groupby('code'):
            amt = float(grp['amount'].sum())
            sh = int(grp['shares'].sum())
            out[code] = {
                'shares': sh, 'amount': amt,
                'fee': float(grp['fee_est'].sum()),
                'price': amt / sh if sh > 0 else 0.0,
                'est_price': float(grp['est_price'].mean()) if 'est_price' in grp.columns and grp['est_price'].notna().any() else None,
            }
        return out

    def build_dim1_candidates(self) -> dict:
        """维度1：候选股 diff（选股层）。"""
        live = self._live_buy_candidates()
        bt = self._bt_snap().get('raw_buy_n_list', []) or []

        live_set, bt_set = set(live), set(bt)
        common = live_set & bt_set
        only_live = live_set - bt_set
        only_bt = bt_set - live_set

        return {
            'live_count': len(live), 'bt_count': len(bt),
            'common': sorted(common),
            'only_live': sorted(only_live),
            'only_bt': sorted(only_bt),
            'match_rate': len(common) / max(len(live), len(bt)) if (live or bt) else 1.0,
        }

    def build_dim2_tradable(self) -> dict:
        """维度2：可交易 diff（合法性层）。"""
        live = self._live_buy_tradable()
        bt = self._bt_snap().get('buy_n_list', []) or []

        live_set, bt_set = set(live), set(bt)
        common = live_set & bt_set
        return {
            'live_count': len(live), 'bt_count': len(bt),
            'common': sorted(common),
            'only_live': sorted(live_set - bt_set),
            'only_bt': sorted(bt_set - live_set),
            'match_rate': len(common) / max(len(live), len(bt)) if (live or bt) else 1.0,
        }

    def build_dim3_orders(self) -> dict:
        """维度3：订单 diff（执行层）。"""
        live_buy = self._live_buy_executed()
        live_sell = self._live_sell_executed()

        bt_snap = self._bt_snap()
        bt_buy_details = {d['code']: d for d in bt_snap.get('executed_buy_details', []) or []}
        bt_sell_details = {d['code']: d for d in bt_snap.get('executed_sell_details', []) or []}

        # 按股票汇总订单差异
        rows = []
        all_codes = set(live_buy) | set(bt_buy_details) | set(live_sell) | set(bt_sell_details)
        for code in sorted(all_codes):
            lb = live_buy.get(code)
            bb = bt_buy_details.get(code)
            ls = live_sell.get(code)
            bs = bt_sell_details.get(code)
            rows.append({
                'code': code,
                'name': _name(code, self.trade_date),
                'live_buy_amount': lb['amount'] if lb else 0.0,
                'bt_buy_amount': float(bb['shares'] * bb['price']) if bb else 0.0,
                'live_sell_amount': ls['amount'] if ls else 0.0,
                'bt_sell_amount': float(bs['shares'] * bs['price']) if bs else 0.0,
            })
        return {'rows': rows,
                'live_buy_total': sum(r['live_buy_amount'] for r in rows),
                'bt_buy_total': sum(r['bt_buy_amount'] for r in rows),
                'live_sell_total': sum(r['live_sell_amount'] for r in rows),
                'bt_sell_total': sum(r['bt_sell_amount'] for r in rows)}

    def build_dim4_slippage(self) -> dict:
        """维度4：滑点 diff（市场冲击层）。实盘的 traded_price vs est_price，回测假定 0。"""
        if self._fills is None or self._fills.empty:
            return {'rows': [], 'avg_slippage': 0.0, 'total_slippage_cost': 0.0}
        df = self._fills.copy()
        if 'slippage_pct' not in df.columns:
            df['slippage_pct'] = None

        rows = []
        total_cost = 0.0
        for _, r in df.iterrows():
            if pd.isna(r.get('slippage_pct')):
                continue
            sp = float(r['slippage_pct'])
            # 滑点成本：买入正滑点=多付，卖出负滑点=少收，绝对值都是成本
            sign = 1 if r['direction'] == 'buy' else -1
            cost = sign * sp / 100 * float(r['amount'])
            rows.append({
                'code': r['code'], 'name': r.get('name', ''),
                'direction': r['direction'],
                'est_price': float(r['est_price']) if not pd.isna(r['est_price']) else None,
                'traded_price': float(r['price']),
                'slippage_pct': sp,
                'amount': float(r['amount']),
                'slippage_cost': cost,
            })
            total_cost += cost
        avg_sp = sum(r['slippage_pct'] for r in rows) / len(rows) if rows else 0.0
        return {'rows': rows, 'avg_slippage': avg_sp, 'total_slippage_cost': total_cost}

    def build_dim5_pnl(self) -> dict:
        """维度5：日 P&L diff（结果层）。逐股对比。"""
        live_pnl: dict[str, dict] = {}
        if self._positions is not None and not self._positions.empty:
            for _, r in self._positions.iterrows():
                live_pnl[r['code']] = {
                    'volume': int(r['volume']),
                    'mv': float(r['market_value']),
                    'daily_pnl': float(r['daily_pnl']) if not pd.isna(r.get('daily_pnl')) else None,
                    'daily_return_pct': float(r['daily_return_pct']) if not pd.isna(r.get('daily_return_pct')) else None,
                }

        # 回测 per-stock（用 _rebuild_backtest_per_stock_pnl 同口径重建）
        bt_pnl = _rebuild_backtest_per_stock_pnl(self._bt) if self._bt else {}

        rows = []
        all_codes = set(live_pnl) | set(bt_pnl)
        for code in sorted(all_codes):
            l = live_pnl.get(code, {})
            b = bt_pnl.get(code, {})
            l_pnl = l.get('daily_pnl')
            b_pnl = b.get('daily_pnl')
            pnl_diff = (l_pnl - b_pnl) if (l_pnl is not None and b_pnl is not None) else None
            l_ret = l.get('daily_return_pct')
            b_ret = b.get('daily_return_pct')
            ret_diff = (l_ret - b_ret) if (l_ret is not None and b_ret is not None) else None
            rows.append({
                'code': code, 'name': _name(code, self.trade_date),
                'live_volume': l.get('volume', 0), 'bt_volume': b.get('volume', 0),
                'live_mv': l.get('mv', 0.0), 'bt_mv': b.get('mv', 0.0),
                'live_daily_pnl': l_pnl, 'bt_daily_pnl': b_pnl,
                'live_daily_return_pct': l_ret, 'bt_daily_return_pct': b_ret,
                'pnl_diff': pnl_diff, 'ret_diff': ret_diff,
            })
        live_valid = [r['live_daily_pnl'] for r in rows if r['live_daily_pnl'] is not None]
        bt_valid = [r['bt_daily_pnl'] for r in rows if r['bt_daily_pnl'] is not None]
        return {'rows': rows,
                'live_total_pnl': sum(live_valid) if live_valid else None,
                'bt_total_pnl': sum(bt_valid) if bt_valid else None}

    # ── 不变量校验 ─────────────────────────────────────────
    # 会计恒等式：sum(per_stock_daily_pnl) == account_daily_pnl
    #     其中 account_daily_pnl = T_asset - Y_asset - net_cash_flow
    # 推导：
    #     asset = cash + market_value
    #     cash_T = cash_Y + net_cf + Σsells - Σbuys - Σfees
    #     mv_T   = Σ(T_lp × T_vol)
    #     asset_T - asset_Y - net_cf
    #       = (Σsells - Σbuys - Σfees) + Σ(T_mv - Y_mv)
    #       = Σ_stock [(T_mv - Y_mv) + sell - buy - fee]
    #       = Σ_stock daily_pnl
    # 这条恒等式必须严格成立——任何漂移都意味着 snapshot/fills/cash_flow 不完整。

    # 容差策略：绝对 ¥1 + 账户 0.05%（万分之五）取大值。
    # 0.05% 来源：cost basis 反推 buy_amt 的精度（QMT avg_price 与真实加权可能微差）
    # + fee 估算偏差累计（按比例放大每股 fee）。
    # 实测 24 只持仓累计尾差约 ¥200 / ¥70 万 ≈ 0.03%。
    _RECONCILE_TOLERANCE_ABS = 1.0
    _RECONCILE_TOLERANCE_PCT = 0.0005

    def _reconcile_tolerance(self, summary: dict) -> float:
        prev = summary.get('prev_asset') or 0
        return max(self._RECONCILE_TOLERANCE_ABS, float(prev) * self._RECONCILE_TOLERANCE_PCT)

    def reconcile_pnl(self, dim5: dict, summary: dict) -> dict:
        """校验 sum(个股 P&L) ≈ 账户 P&L 不变量。

        Returns:
            {'per_stock_pnl_sum': float|None,
             'account_pnl': float|None,
             'diff': float|None,                 # per_stock_sum - account
             'tolerance': float,                  # 当前容差
             'unreconcilable_codes': list[str],
             'within_tolerance': bool}
        """
        per_stock = dim5.get('live_total_pnl')
        account = summary.get('live_daily_pnl')
        unrec = [r['code'] for r in dim5.get('rows', [])
                 if r.get('live_daily_pnl') is None]
        tolerance = self._reconcile_tolerance(summary)
        diff = None
        within = True
        if per_stock is not None and account is not None:
            diff = per_stock - account
            within = abs(diff) <= tolerance
            if not within:
                trading_logger.warning(
                    f"[PostClose 校验] 个股日 P&L 总和 ¥{per_stock:+,.2f} "
                    f"!= 账户日变化 ¥{account:+,.2f}, 差 ¥{diff:+,.2f} (容差 ±¥{tolerance:.0f})。"
                    f"无法计算 P&L 的股票: {len(unrec)} 只 {unrec[:5]}。"
                    f"常见原因: 昨日持仓 snapshot 缺失 / fills 漏记 / cash_flow 未同步。"
                )
        return {
            'per_stock_pnl_sum': per_stock,
            'account_pnl': account,
            'diff': diff,
            'tolerance': tolerance,
            'unreconcilable_codes': unrec,
            'within_tolerance': within,
        }

    def build_summary(self) -> dict:
        """整体汇总（账户层）。"""
        bt_snap = self._bt_snap()
        bt_total_asset = bt_snap.get('total_asset', 0.0) if bt_snap else 0.0
        bt_daily_ret = bt_snap.get('daily_return_pct', 0.0) if bt_snap else 0.0

        # 实盘账户层日收益
        live_daily_pnl = None
        live_daily_ret = None
        if self._prev_asset and self._asset and self._prev_asset > 0:
            live_daily_pnl = self._asset - self._prev_asset - self._net_cash_flow
            live_daily_ret = live_daily_pnl / self._prev_asset * 100

        bt_daily_pnl = None
        if self._prev_asset and self._prev_asset > 0:
            bt_daily_pnl = self._prev_asset * bt_daily_ret / 100

        return {
            'live_total_asset': self._asset,
            'bt_total_asset': bt_total_asset,
            'live_daily_pnl': live_daily_pnl,
            'bt_daily_pnl': bt_daily_pnl,
            'live_daily_return_pct': live_daily_ret,
            'bt_daily_return_pct': bt_daily_ret,
            'net_cash_flow': self._net_cash_flow,
            'prev_asset': self._prev_asset,
        }

    def build(self) -> dict:
        """组装报告数据。

        只输出 3 个有实际诊断价值的维度：
          - dim3_orders   操作差距（多退少补的实盘 vs 回测）
          - dim4_slippage 滑点（成交价 vs 开盘价，HTML debug 用）
          - dim5_pnl      逐股盈亏对比
          - summary       收益对比
          - reconcile     账务自检

        候选股层 / 合法性层（dim1 / dim2）不再输出：
        实盘和回测共用同一个 select_topn / batch_limit_check，结果必然一致；
        如出现差异即 bug，由单元测试覆盖，不在日常报告里耗注意力。
        """
        dim5 = self.build_dim5_pnl()
        summary = self.build_summary()
        return {
            'date': self.trade_date.isoformat(),
            'dim3_orders': self.build_dim3_orders(),
            'dim4_slippage': self.build_dim4_slippage(),
            'dim5_pnl': dim5,
            'summary': summary,
            'reconcile': self.reconcile_pnl(dim5, summary),
        }

    # ── 飞书卡片 (JSON Schema 2.0 + 原生 table 组件) ───────────────
    #  设计原则：
    #    1. 所有股票只显示名称，不显示代码
    #    2. 所有 diff 列遵循「实盘 | 回测 | 差」三列对账格式
    #    3. 用飞书 v2 schema 的原生 `table` 组件（不是 code-block hack）
    #    4. 不做 Top N 截断，page_size=10 自带分页

    def _diff_md(self, v: float | None) -> str:
        """正红负绿（中国股票配色），用于 diff 列。"""
        if v is None or pd.isna(v):
            return '-'
        s = _fmt_diff_money(v)
        if v > 0:
            return f'<font color="red">{s}</font>'
        if v < 0:
            return f'<font color="green">{s}</font>'
        return s

    def _diff_pct_md(self, v: float | None) -> str:
        if v is None or pd.isna(v):
            return '-'
        s = _fmt_pct(v, sign=True)
        if v > 0:
            return f'<font color="red">{s}</font>'
        if v < 0:
            return f'<font color="green">{s}</font>'
        return s

    def _card_header(self, data: dict) -> dict:
        """卡片头：一眼看到「实盘 vs 回测」的核心差距。
        颜色判定按 P&L 差额相对账户总值的占比：
          < 0.1%  绿  几乎对齐
          < 0.5%  黄  有偏差
          >=0.5%  红  显著差异
        """
        s = data['summary']
        live_pnl = s['live_daily_pnl'] or 0
        bt_pnl = s['bt_daily_pnl'] or 0
        diff = live_pnl - bt_pnl
        base = (s.get('prev_asset') or 1_000_000)  # 用 prev_asset 做基数避免除零
        pct = abs(diff) / base
        if pct < 0.001:
            template = 'green'
        elif pct < 0.005:
            template = 'orange'
        else:
            template = 'red'

        # 副标题：直接说"实盘比回测多挣/少挣 ¥X"
        sign_word = '多挣' if diff > 0 else ('少挣' if diff < 0 else '持平')
        return {
            'title': {'tag': 'plain_text',
                      'content': f"盘后对账 {self.trade_date.isoformat()}"},
            'subtitle': {'tag': 'plain_text',
                         'content': f"实盘比回测{sign_word} {_fmt_diff_money(abs(diff))} "
                                    f"({_fmt_pct(abs(diff)/base*100, sign=False)})"},
            'template': template,
        }

    def _v2_table(self, *, title: str, columns: list[dict], rows: list[dict],
                  element_id: str, page_size: int = 10,
                  freeze_first_column: bool = True) -> list[dict]:
        """生成 [title_div, table] 两个 v2 组件。"""
        return [
            {'tag': 'div', 'text': {'tag': 'lark_md', 'content': title}},
            {
                'tag': 'table',
                'element_id': element_id,
                'page_size': page_size,
                'row_height': 'low',
                'freeze_first_column': freeze_first_column,
                'header_style': {'text_align': 'center', 'bold': True,
                                 'background_style': 'grey', 'text_color': 'grey'},
                'columns': columns,
                'rows': rows,
            },
        ]

    def _v2_summary_table(self, data: dict) -> list[dict]:
        """收益对比：3 行（日 P&L / 日收益率 / 总资产），实盘 / 回测 / 差。"""
        s = data['summary']

        def _ad(a, b):
            return (a - b) if (a is not None and b is not None) else None

        rows = [
            {'indicator': '今日盈亏',
             'live': _fmt_diff_money(s['live_daily_pnl']),
             'bt': _fmt_diff_money(s['bt_daily_pnl']),
             'diff': self._diff_md(_ad(s['live_daily_pnl'], s['bt_daily_pnl']))},
            {'indicator': '今日收益率',
             'live': _fmt_pct(s['live_daily_return_pct'], sign=True),
             'bt': _fmt_pct(s['bt_daily_return_pct'], sign=True),
             'diff': self._diff_pct_md(_ad(s['live_daily_return_pct'], s['bt_daily_return_pct']))},
            {'indicator': '总资产',
             'live': _fmt_money(s['live_total_asset']),
             'bt': _fmt_money(s['bt_total_asset']),
             'diff': self._diff_md(_ad(s['live_total_asset'], s['bt_total_asset']))},
        ]
        columns = [
            {'name': 'indicator', 'display_name': '', 'data_type': 'lark_md',
             'horizontal_align': 'left'},
            {'name': 'live', 'display_name': '实盘', 'data_type': 'lark_md',
             'horizontal_align': 'right'},
            {'name': 'bt', 'display_name': '回测', 'data_type': 'lark_md',
             'horizontal_align': 'right'},
            {'name': 'diff', 'display_name': '差额', 'data_type': 'lark_md',
             'horizontal_align': 'right'},
        ]
        return self._v2_table(title='**📈 收益对比**', columns=columns, rows=rows,
                              element_id='summary_tbl', page_size=10,
                              freeze_first_column=False)

    def _v2_dim3_table(self, data: dict) -> list[dict]:
        """操作差距：每只股票的「净买入」金额，实盘 vs 回测对比。
        净买入 = 买入金额 - 卖出金额。正=买入(少补)，负=卖出(多退)。
        全量展示，按 |实盘净买 - 回测净买| 降序，合计行放最后。
        """
        d3 = data['dim3_orders']

        def _net(r, side: str) -> float:
            return r[f'{side}_buy_amount'] - r[f'{side}_sell_amount']

        all_rows = sorted(
            [r for r in d3['rows']
             if r['live_buy_amount'] or r['bt_buy_amount']
             or r['live_sell_amount'] or r['bt_sell_amount']],
            key=lambda r: abs(_net(r, 'live') - _net(r, 'bt')),
            reverse=True,
        )

        table_rows = []
        for r in all_rows:
            live_net = _net(r, 'live')
            bt_net = _net(r, 'bt')
            table_rows.append({
                'name': self._name_of(r['code']),
                'live': self._diff_md(live_net),
                'bt': self._diff_md(bt_net),
                'diff': self._diff_md(live_net - bt_net),
            })
        live_total_net = d3['live_buy_total'] - d3['live_sell_total']
        bt_total_net = d3['bt_buy_total'] - d3['bt_sell_total']
        table_rows.append({
            'name': '— 合计 —',
            'live': self._diff_md(live_total_net),
            'bt': self._diff_md(bt_total_net),
            'diff': self._diff_md(live_total_net - bt_total_net),
        })

        columns = [
            {'name': 'name', 'display_name': '股票', 'data_type': 'lark_md',
             'horizontal_align': 'left'},
            {'name': 'live', 'display_name': '实盘净买', 'data_type': 'lark_md',
             'horizontal_align': 'right'},
            {'name': 'bt', 'display_name': '回测净买', 'data_type': 'lark_md',
             'horizontal_align': 'right'},
            {'name': 'diff', 'display_name': '差额', 'data_type': 'lark_md',
             'horizontal_align': 'right'},
        ]
        return self._v2_table(
            title=(f"**🔀 操作差距** · {len(all_rows)} 只股票 "
                   f"·  +少补 / −多退"),
            columns=columns, rows=table_rows,
            element_id='dim3_tbl', page_size=20)

    def _v2_dim5_table(self, data: dict) -> list[dict]:
        """逐股盈亏对比：每只股票的今日盈亏，实盘 vs 回测。
        全量展示，按 |差额| 降序，合计行放最后。
        """
        rows = data['dim5_pnl']['rows']
        live_total = data['dim5_pnl']['live_total_pnl']
        bt_total = data['dim5_pnl']['bt_total_pnl']

        if not rows:
            return [{'tag': 'div', 'text': {'tag': 'lark_md',
                                            'content': '**💹 逐股盈亏对比** · 今日无持仓'}}]

        rows_sorted = sorted(
            rows,
            key=lambda r: (r['pnl_diff'] is None,
                           -abs(r['pnl_diff']) if r['pnl_diff'] is not None else 0),
        )

        table_rows = []
        for r in rows_sorted:
            nm = self._name_of(r['code'])
            table_rows.append({
                'name': nm,
                'live_pnl': self._diff_md(r['live_daily_pnl']),
                'bt_pnl': self._diff_md(r['bt_daily_pnl']),
                'diff': self._diff_md(r['pnl_diff']),
                'live_ret': self._diff_pct_md(r['live_daily_return_pct']),
                'bt_ret': self._diff_pct_md(r['bt_daily_return_pct']),
            })
        total_diff = (live_total - bt_total) if (live_total is not None and bt_total is not None) else None
        table_rows.append({
            'name': '— 合计 —',
            'live_pnl': self._diff_md(live_total),
            'bt_pnl': self._diff_md(bt_total),
            'diff': self._diff_md(total_diff),
            'live_ret': '-', 'bt_ret': '-',
        })

        columns = [
            {'name': 'name', 'display_name': '股票', 'data_type': 'lark_md',
             'horizontal_align': 'left'},
            {'name': 'live_pnl', 'display_name': '实盘盈亏', 'data_type': 'lark_md',
             'horizontal_align': 'right'},
            {'name': 'bt_pnl', 'display_name': '回测盈亏', 'data_type': 'lark_md',
             'horizontal_align': 'right'},
            {'name': 'diff', 'display_name': '差额', 'data_type': 'lark_md',
             'horizontal_align': 'right'},
            {'name': 'live_ret', 'display_name': '实盘涨幅', 'data_type': 'lark_md',
             'horizontal_align': 'right'},
            {'name': 'bt_ret', 'display_name': '回测涨幅', 'data_type': 'lark_md',
             'horizontal_align': 'right'},
        ]
        return self._v2_table(
            title=f"**💹 逐股盈亏对比** · {len(rows)} 只持仓",
            columns=columns, rows=table_rows,
            element_id='dim5_tbl', page_size=20)

    def _v2_footer(self, data: dict, html_path: Optional[Path]) -> dict:
        """卡片底部：账务自检 + HTML 附件路径。
        
        候选股层 / 合法性层不校验：实盘和回测共用同一个 select_topn / batch_limit_check
        函数，差异只可能是 bug（用单元测试覆盖），不在日常报告里展示。
        """
        parts = []

        # 1. 账务自检（逐股盈亏总和 应 ≈ 账户日变化）
        rec = data.get('reconcile') or {}
        if rec.get('per_stock_pnl_sum') is not None and rec.get('account_pnl') is not None:
            within = rec.get('within_tolerance')
            tol = rec.get('tolerance', 1)
            diff = rec['diff']
            if within:
                parts.append(
                    f"<font color=\"grey\">✓ 账务自检通过（逐股盈亏总和 - 账户日变化 = "
                    f"{_fmt_diff_money(diff)}，在容差 ±¥{tol:.0f} 内）</font>"
                )
            else:
                parts.append(
                    f"<font color=\"red\">✗ 账务自检异常：逐股盈亏总和 - 账户日变化 = "
                    f"{_fmt_diff_money(diff)}（超容差 ±¥{tol:.0f}）</font>"
                )

        # 2. HTML 附件
        if html_path:
            parts.append(f"<font color=\"grey\">完整明细见附件：{html_path.name}</font>")
        else:
            parts.append('<font color="grey">完整明细见 HTML 附件</font>')

        return {'tag': 'div', 'text': {'tag': 'lark_md', 'content': '\n\n'.join(parts)}}

    def send(self, html_path: Optional[Path] = None, attach_html: bool = True):
        """发送飞书 v2 卡片 + 上传 HTML 附件。

        卡片结构（精简版）：
          ┌─ 标题：实盘比回测多挣/少挣 ¥X
          ├─ 📈 收益对比（盈亏 / 涨幅 / 总资产 三行）
          ├─ 🔀 操作差距（每只股票的买卖差，按差额降序）
          ├─ 💹 逐股盈亏对比（每只股票的盈亏差，按差额降序）
          └─ ⚠ 候选股 / 合法性偏差提示 + ✓ 账务自检 + HTML 附件链接
        """
        data = self.build()
        elements = []
        elements.extend(self._v2_summary_table(data))
        elements.append({'tag': 'hr'})
        elements.extend(self._v2_dim3_table(data))
        elements.append({'tag': 'hr'})
        elements.extend(self._v2_dim5_table(data))
        elements.append({'tag': 'hr'})
        elements.append(self._v2_footer(data, html_path))

        card = {
            'schema': '2.0',
            'config': {'update_multi': True},
            'header': self._card_header(data),
            'body': {'elements': elements},
        }
        try:
            lark_sender.send_card(card)
        except Exception as e:
            trading_logger.warning(f"[PostClose] 飞书卡片失败: {e}")

        # HTML 附件上传
        if attach_html and html_path and html_path.exists():
            try:
                lark_sender.send_file(str(html_path),
                                      file_name=f"diff_{self.trade_date.isoformat()}.html")
            except Exception as e:
                trading_logger.warning(f"[PostClose] 飞书 HTML 附件失败: {e}")

        s = data['summary']
        live = s.get('live_daily_pnl') or 0
        bt = s.get('bt_daily_pnl') or 0
        trading_logger.info(
            f"[PostClose] {self.trade_date.isoformat()} · "
            f"实盘盈亏 {_fmt_diff_money(live)} · 回测盈亏 {_fmt_diff_money(bt)} · "
            f"差 {_fmt_diff_money(live - bt)}"
        )
        return data

    # ── HTML 报告 ───────────────────────────────────────────

    def _reconcile_html(self, rec: dict) -> str:
        """对账校验区块的 HTML 片段。"""
        if not rec or rec.get('per_stock_pnl_sum') is None or rec.get('account_pnl') is None:
            return (
                "<div class='summary' style='border-left: 4px solid #8899a6;'>"
                "<div class='label'>🔍 账务对账校验</div>"
                "<div>缺少数据，无法校验 ∑(个股 P&L) vs 账户日变化</div>"
                "</div>"
            )
        diff = rec['diff']
        within = rec.get('within_tolerance', True)
        unrec = rec.get('unreconcilable_codes') or []
        color = '#089981' if within else '#f23645'
        status = '✓ 通过' if within else '✗ 偏差超阈值'
        unrec_html = (
            f"<div>无法计算 P&L 的股票（{len(unrec)} 只）: {', '.join(unrec[:20])}{'…' if len(unrec) > 20 else ''}</div>"
            if unrec else ''
        )
        return (
            f"<div class='summary' style='border-left: 4px solid {color};'>"
            f"<div class='label'>🔍 账务对账校验 — <strong style='color:{color}'>{status}</strong></div>"
            f"<div>∑(个股日 P&L) = {_fmt_diff_money(rec['per_stock_pnl_sum'])}, "
            f"账户日 P&L = {_fmt_diff_money(rec['account_pnl'])}, "
            f"差 = <strong style='color:{color}'>{_fmt_diff_money(diff)}</strong> "
            f"(容差 ±¥{rec.get('tolerance', 1):.0f})</div>"
            f"{unrec_html}"
            f"</div>"
        )

    def to_html(self, out_path: Path):
        data = self.build()
        html = self._build_html(data)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding='utf-8')
        trading_logger.info(f"[PostClose] HTML 报告: {out_path}")
        return out_path

    def _build_html(self, data: dict) -> str:
        s = data['summary']
        d3, d4, d5 = data['dim3_orders'], data['dim4_slippage'], data['dim5_pnl']

        def _orders_table(rows):
            html = ['<table><thead><tr><th>代码</th><th>名称</th>',
                    '<th>买入实盘</th><th>买入回测</th>',
                    '<th>卖出实盘</th><th>卖出回测</th></tr></thead><tbody>']
            for r in rows:
                html.append(
                    f"<tr><td>{r['code']}</td><td>{r['name']}</td>"
                    f"<td>{_fmt_money(r['live_buy_amount'])}</td>"
                    f"<td>{_fmt_money(r['bt_buy_amount'])}</td>"
                    f"<td>{_fmt_money(r['live_sell_amount'])}</td>"
                    f"<td>{_fmt_money(r['bt_sell_amount'])}</td></tr>"
                )
            html.append('</tbody></table>')
            return '\n'.join(html)

        def _slippage_table(rows):
            html = ['<table><thead><tr><th>代码</th><th>方向</th><th>est_price</th>',
                    '<th>traded_price</th><th>滑点 %</th><th>滑点成本</th></tr></thead><tbody>']
            for r in rows:
                cls = 'positive' if r['slippage_cost'] > 0 else 'negative'
                html.append(
                    f"<tr><td>{r['code']} {r['name']}</td><td>{r['direction']}</td>"
                    f"<td>{r['est_price']:.3f}</td><td>{r['traded_price']:.3f}</td>"
                    f"<td class='{cls}'>{r['slippage_pct']:+.3f}%</td>"
                    f"<td class='{cls}'>{_fmt_diff_money(r['slippage_cost'])}</td></tr>"
                )
            html.append('</tbody></table>')
            return '\n'.join(html)

        def _pnl_table(rows):
            html = ['<table><thead><tr><th>代码</th><th>名称</th>',
                    '<th>仓位实盘</th><th>仓位回测</th>',
                    '<th>市值实盘</th><th>市值回测</th>',
                    '<th>P&L 实盘</th><th>P&L 回测</th><th>P&L 差</th>',
                    '<th>收益 实盘</th><th>收益 回测</th><th>收益差</th></tr></thead><tbody>']
            for r in sorted(rows, key=lambda x: abs(x['pnl_diff']) if x['pnl_diff'] is not None else 0, reverse=True):
                pd_cls = 'positive' if (r['pnl_diff'] or 0) > 0 else ('negative' if (r['pnl_diff'] or 0) < 0 else '')
                html.append(
                    f"<tr><td>{r['code']}</td><td>{r['name']}</td>"
                    f"<td>{r['live_volume']}</td><td>{r['bt_volume']}</td>"
                    f"<td>{_fmt_money(r['live_mv'])}</td>"
                    f"<td>{_fmt_money(r['bt_mv'])}</td>"
                    f"<td>{_fmt_diff_money(r['live_daily_pnl'])}</td>"
                    f"<td>{_fmt_diff_money(r['bt_daily_pnl'])}</td>"
                    f"<td class='{pd_cls}'>{_fmt_diff_money(r['pnl_diff'])}</td>"
                    f"<td>{_fmt_pct(r['live_daily_return_pct'], sign=True)}</td>"
                    f"<td>{_fmt_pct(r['bt_daily_return_pct'], sign=True)}</td>"
                    f"<td class='{pd_cls}'>{_fmt_pct(r['ret_diff'], sign=True)}</td>"
                    f"</tr>"
                )
            html.append('</tbody></table>')
            return '\n'.join(html)

        return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>盘后 Diff {data['date']}</title>
<style>
  body {{ font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
          background: #0f1419; color: #e1e8ed; max-width: 1400px; margin: 0 auto; padding: 24px; }}
  h1, h2 {{ color: #ffffff; border-bottom: 1px solid #2c3e50; padding-bottom: 8px; }}
  .summary {{ background: #1a2332; padding: 16px; border-radius: 8px; margin-bottom: 24px; }}
  .summary .metric {{ display: inline-block; margin-right: 32px; }}
  .summary .label {{ color: #8899a6; font-size: 12px; }}
  .summary .value {{ font-size: 20px; font-weight: 600; }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0 24px; }}
  th {{ background: #1a2332; padding: 10px; text-align: right; font-size: 13px; color: #8899a6; }}
  th:first-child, th:nth-child(2) {{ text-align: left; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #2c3e50; text-align: right; font-size: 13px; }}
  td:first-child, td:nth-child(2) {{ text-align: left; font-family: monospace; }}
  .positive {{ color: #f23645; }}
  .negative {{ color: #089981; }}
  .section {{ margin-bottom: 32px; }}
  .codes {{ background: #1a2332; padding: 12px; border-radius: 4px; margin-bottom: 8px;
            font-family: monospace; font-size: 13px; line-height: 1.6; }}
  .match-rate {{ font-size: 24px; font-weight: 700; }}
  .match-good {{ color: #089981; }}
  .match-bad {{ color: #f23645; }}
</style></head><body>
<h1>盘后 Diff 报告 — {data['date']}</h1>

<div class='summary'>
  <div class='metric'><div class='label'>实盘总资产</div><div class='value'>{_fmt_money(s['live_total_asset'])}</div></div>
  <div class='metric'><div class='label'>回测总资产</div><div class='value'>{_fmt_money(s['bt_total_asset'])}</div></div>
  <div class='metric'><div class='label'>实盘日 P&L</div><div class='value'>{_fmt_diff_money(s['live_daily_pnl'])} ({_fmt_pct(s['live_daily_return_pct'], sign=True)})</div></div>
  <div class='metric'><div class='label'>回测日 P&L</div><div class='value'>{_fmt_diff_money(s['bt_daily_pnl'])} ({_fmt_pct(s['bt_daily_return_pct'], sign=True)})</div></div>
</div>
{self._reconcile_html(data.get('reconcile') or {})}

<div class='section'>
  <h2>操作差距（多退少补：实盘 vs 回测）</h2>
  <div>买入: 实盘 {_fmt_money(d3['live_buy_total'])} · 回测 {_fmt_money(d3['bt_buy_total'])}</div>
  <div>卖出: 实盘 {_fmt_money(d3['live_sell_total'])} · 回测 {_fmt_money(d3['bt_sell_total'])}</div>
  {_orders_table(d3['rows'])}
</div>

<div class='section'>
  <h2>滑点明细（成交价 vs 开盘价，仅 debug 用）</h2>
  <div>平均滑点 {d4['avg_slippage']:+.3f}% · 总滑点成本 {_fmt_diff_money(d4['total_slippage_cost'])}</div>
  {_slippage_table(d4['rows'])}
</div>

<div class='section'>
  <h2>逐股盈亏对比</h2>
  <div>实盘合计 {_fmt_diff_money(d5['live_total_pnl'])} · 回测合计 {_fmt_diff_money(d5['bt_total_pnl'])}</div>
  {_pnl_table(d5['rows'])}
</div>

</body></html>"""
