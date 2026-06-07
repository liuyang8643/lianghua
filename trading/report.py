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
from trading.lark.format import fmt_money, fmt_pct, fmt_diff_money
from trading.lark.sender import lark_sender, LarkMsgLevel
from trading.logger import trading_logger

_TRADE_DIR = Path(__file__).resolve().parents[1] / "data" / "live_trades"


def _name(code: str, signal_date: date) -> str:
    return get_stock_name_at_date(code, signal_date) or ''


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

        # 按股票汇总订单差异（金额 + 股数两个口径；股数差直接反映「量缺口」）
        rows = []
        all_codes = set(live_buy) | set(bt_buy_details) | set(live_sell) | set(bt_sell_details)
        for code in sorted(all_codes):
            lb = live_buy.get(code)
            bb = bt_buy_details.get(code)
            ls = live_sell.get(code)
            bs = bt_sell_details.get(code)
            live_buy_sh = int(lb['shares']) if lb else 0
            bt_buy_sh = int(bb['shares']) if bb else 0
            live_sell_sh = int(ls['shares']) if ls else 0
            bt_sell_sh = int(bs['shares']) if bs else 0
            rows.append({
                'code': code,
                'name': _name(code, self.trade_date),
                'live_buy_amount': lb['amount'] if lb else 0.0,
                'bt_buy_amount': float(bb['shares'] * bb['price']) if bb else 0.0,
                'live_sell_amount': ls['amount'] if ls else 0.0,
                'bt_sell_amount': float(bs['shares'] * bs['price']) if bs else 0.0,
                'live_buy_shares': live_buy_sh,
                'bt_buy_shares': bt_buy_sh,
                'live_sell_shares': live_sell_sh,
                'bt_sell_shares': bt_sell_sh,
                # 量缺口：>0 实盘买少了 / 卖多了；净缺口聚焦买入是否补满
                'buy_shares_gap': bt_buy_sh - live_buy_sh,
                'sell_shares_gap': bt_sell_sh - live_sell_sh,
            })
        return {'rows': rows,
                'live_buy_total': sum(r['live_buy_amount'] for r in rows),
                'bt_buy_total': sum(r['bt_buy_amount'] for r in rows),
                'live_sell_total': sum(r['live_sell_amount'] for r in rows),
                'bt_sell_total': sum(r['bt_sell_amount'] for r in rows),
                'buy_shares_gap_total': sum(r['buy_shares_gap'] for r in rows),
                'sell_shares_gap_total': sum(r['sell_shares_gap'] for r in rows)}

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
            l_present, b_present = code in live_pnl, code in bt_pnl
            l_pnl = l.get('daily_pnl')
            b_pnl = b.get('daily_pnl')
            l_ret = l.get('daily_return_pct')
            b_ret = b.get('daily_return_pct')
            # 一边完全没持有/没交易该票 → 它对那边当日盈亏贡献为 0(而非"未知")。
            # 这样单边持有的票(如回测买进、实盘没买进)的差额能正确体现,逐行差额可加总到合计;
            # 仅当某边「确实持有该票却算不出 P&L」时才记为未知(None)。
            l_eff = l_pnl if l_present else 0.0
            b_eff = b_pnl if b_present else 0.0
            pnl_diff = (l_eff - b_eff) if (l_eff is not None and b_eff is not None) else None
            l_ret_eff = l_ret if l_present else 0.0
            b_ret_eff = b_ret if b_present else 0.0
            ret_diff = (l_ret_eff - b_ret_eff) if (l_ret_eff is not None and b_ret_eff is not None) else None
            rows.append({
                'code': code, 'name': _name(code, self.trade_date),
                'live_volume': l.get('volume', 0), 'bt_volume': b.get('volume', 0),
                'live_mv': l.get('mv', 0.0), 'bt_mv': b.get('mv', 0.0),
                'live_daily_pnl': l_pnl, 'bt_daily_pnl': b_pnl,
                'live_daily_return_pct': l_ret, 'bt_daily_return_pct': b_ret,
                'pnl_diff': pnl_diff, 'ret_diff': ret_diff,
                '_l_eff': l_eff, '_b_eff': b_eff,  # 合计用(未持有计 0,持有却算不出计 None)
            })
        live_valid = [r['_l_eff'] for r in rows if r['_l_eff'] is not None]
        bt_valid = [r['_b_eff'] for r in rows if r['_b_eff'] is not None]
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
        # 用账户层日变化（asset - prev - net_cash_flow）做对账基准。
        # 注意 summary['live_daily_pnl'] 现已切换为「个股口径」，不能再拿它对账，
        # 否则会自己跟自己比恒为 0；这里必须取独立的 live_account_pnl。
        account = summary.get('live_account_pnl')
        # 只把「实盘确实持有(volume>0)却算不出 P&L」的票算作无法对账;
        # 实盘根本没持有的票贡献为 0、可对账,不应计入。
        unrec = [r['code'] for r in dim5.get('rows', [])
                 if r.get('live_daily_pnl') is None and int(r.get('live_volume', 0) or 0) > 0]
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

    def build_summary(self, dim5: dict | None = None) -> dict:
        """整体汇总（账户层）。

        「今日盈亏」口径：优先取**个股日 P&L 总和**（dim5.live_total_pnl）。
        个股口径 = Σ[(T市值-Y市值)+卖出-买入-费用]，天然免疫未记账的银证出入金，
        故作为权威总盈亏。仅当存在「持有却算不出 P&L」的持仓（疑似漏记成交）时，
        个股总和不可信，回退账户层 (asset - prev_asset - net_cash_flow)。
        账户层数值仍保留在 live_account_pnl，供 reconcile 对账与异常告警使用。
        """
        bt_snap = self._bt_snap()
        bt_total_asset = bt_snap.get('total_asset', 0.0) if bt_snap else 0.0
        bt_daily_ret = bt_snap.get('daily_return_pct', 0.0) if bt_snap else 0.0

        # 账户层日变化（含未剔除的出入金风险）
        live_account_pnl = None
        if self._prev_asset and self._asset and self._prev_asset > 0:
            live_account_pnl = self._asset - self._prev_asset - self._net_cash_flow

        # 个股口径：仅当所有持仓都能算出 P&L（无 unreconcilable）时才采信
        per_stock_pnl = None
        if dim5 is not None:
            unrec = [r for r in dim5.get('rows', [])
                     if r.get('live_daily_pnl') is None and int(r.get('live_volume', 0) or 0) > 0]
            if not unrec:
                per_stock_pnl = dim5.get('live_total_pnl')

        live_daily_pnl = per_stock_pnl if per_stock_pnl is not None else live_account_pnl
        live_pnl_source = 'per_stock' if per_stock_pnl is not None else 'account'
        live_daily_ret = None
        if live_daily_pnl is not None and self._prev_asset and self._prev_asset > 0:
            live_daily_ret = live_daily_pnl / self._prev_asset * 100

        bt_daily_pnl = None
        if self._prev_asset and self._prev_asset > 0:
            bt_daily_pnl = self._prev_asset * bt_daily_ret / 100

        return {
            'live_total_asset': self._asset,
            'bt_total_asset': bt_total_asset,
            'live_daily_pnl': live_daily_pnl,
            'live_account_pnl': live_account_pnl,
            'live_pnl_source': live_pnl_source,
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
        summary = self.build_summary(dim5)
        return {
            'date': self.trade_date.isoformat(),
            'dim3_orders': self.build_dim3_orders(),
            'dim4_slippage': self.build_dim4_slippage(),
            'dim5_pnl': dim5,
            'summary': summary,
            'reconcile': self.reconcile_pnl(dim5, summary),
        }

    def send(self, html_path: Optional[Path] = None, attach_html: bool = True):
        """上传 HTML 附件 + 对账异常告警。返回 report data 供调用方取 P&L。

        不再发独立飞书日报卡片（统一由 day_board 单卡全天更新）。
        """
        data = self.build()

        # HTML 附件上传（不再发独立飞书卡片，日报统一由 day_board 发）
        if attach_html and html_path and html_path.exists():
            try:
                lark_sender.send_file(str(html_path),
                                      file_name=f"diff_{self.trade_date.isoformat()}.html")
            except Exception as e:
                trading_logger.warning(f"[PostClose] 飞书 HTML 附件失败: {e}")

        # 对账异常 → 单独推送告警卡片（残差/疑似出入金，待人工确认）
        self._maybe_alert_reconcile(data)

        s = data['summary']
        live = s.get('live_daily_pnl') or 0
        bt = s.get('bt_daily_pnl') or 0
        trading_logger.info(
            f"[PostClose] {self.trade_date.isoformat()} · "
            f"实盘盈亏 {fmt_diff_money(live)} · 回测盈亏 {fmt_diff_money(bt)} · "
            f"差 {fmt_diff_money(live - bt)}"
        )
        return data

    def _maybe_alert_reconcile(self, data: dict):
        """账户日盈亏 ≠ 个股盈亏总和（超容差）时，飞书推送残差告警，待人工确认是否出入金。

        残差 diff = Σ个股 - 账户；疑似未记账净出入金 = 账户 - Σ个股 = -diff
        （>0 入金 / <0 出金）。报告「今日盈亏」已以个股口径为准，此处只做提示。
        """
        rec = (data or {}).get('reconcile') or {}
        if rec.get('within_tolerance', True):
            return
        diff = rec.get('diff')
        per_stock = rec.get('per_stock_pnl_sum')
        account = rec.get('account_pnl')
        if diff is None or per_stock is None or account is None:
            return
        unrec = rec.get('unreconcilable_codes') or []
        tol = rec.get('tolerance', 0.0)

        if unrec:
            sub = "盈亏对账异常 · 疑似漏记成交"
            content = (
                f"账户日盈亏 **{fmt_diff_money(account)}** 与个股盈亏总和 "
                f"**{fmt_diff_money(per_stock)}** 不一致，残差 **{fmt_diff_money(diff)}** "
                f"(容差 ±¥{tol:.0f})。\n"
                f"⚠️ 有 {len(unrec)} 只持仓无法计算盈亏（疑似漏记成交）："
                f"{', '.join(self._name_of(c) for c in unrec[:10])}。\n"
                f"残差可能来自漏记成交而非出入金，请先人工核查成交记录。"
            )
        else:
            suspected = account - per_stock  # 疑似未记账净出入金
            direction = '入金' if suspected >= 0 else '出金'
            sub = "盈亏对账异常 · 疑似未记录出入金"
            content = (
                f"账户日盈亏 **{fmt_diff_money(account)}**（按总资产变化推算），"
                f"个股盈亏总和 **{fmt_diff_money(per_stock)}**，"
                f"残差 **{fmt_diff_money(diff)}** (容差 ±¥{tol:.0f})。\n"
                f"💰 疑似当日存在未记录的净{direction} ≈ **¥{abs(suspected):,.0f}**。\n"
                f"报告「今日盈亏」已采用个股口径（{fmt_diff_money(per_stock)}）为准。\n"
                f"请确认当日是否有银证转账；若确有出入金，可手动补记账以消除该提示。"
            )
        try:
            lark_sender.send_notification_card(
                content=content, level=LarkMsgLevel.Warning,
                title=f"⚠️ 盈亏对账异常 @ {self.trade_date.isoformat()}",
                sub_title=sub,
            )
            trading_logger.info(
                f"[PostClose] 已推送对账异常告警: 残差 {fmt_diff_money(diff)} "
                f"(账户 {fmt_diff_money(account)} vs 个股 {fmt_diff_money(per_stock)})"
            )
        except Exception as e:
            trading_logger.warning(f"[PostClose] 对账异常通知失败: {e}")

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
            f"<div>∑(个股日 P&L) = {fmt_diff_money(rec['per_stock_pnl_sum'])}, "
            f"账户日 P&L = {fmt_diff_money(rec['account_pnl'])}, "
            f"差 = <strong style='color:{color}'>{fmt_diff_money(diff)}</strong> "
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
                    '<th>买股实盘</th><th>买股回测</th><th>买量缺口</th>',
                    '<th>卖出实盘</th><th>卖出回测</th></tr></thead><tbody>']
            for r in rows:
                gap = r.get('buy_shares_gap', 0)
                gap_cls = 'positive' if gap > 0 else ('negative' if gap < 0 else '')
                html.append(
                    f"<tr><td>{r['code']}</td><td>{r['name']}</td>"
                    f"<td>{fmt_money(r['live_buy_amount'])}</td>"
                    f"<td>{fmt_money(r['bt_buy_amount'])}</td>"
                    f"<td>{r.get('live_buy_shares', 0)}</td>"
                    f"<td>{r.get('bt_buy_shares', 0)}</td>"
                    f"<td class='{gap_cls}'>{gap:+d}</td>"
                    f"<td>{fmt_money(r['live_sell_amount'])}</td>"
                    f"<td>{fmt_money(r['bt_sell_amount'])}</td></tr>"
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
                    f"<td class='{cls}'>{fmt_diff_money(r['slippage_cost'])}</td></tr>"
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
                    f"<td>{fmt_money(r['live_mv'])}</td>"
                    f"<td>{fmt_money(r['bt_mv'])}</td>"
                    f"<td>{fmt_diff_money(r['live_daily_pnl'])}</td>"
                    f"<td>{fmt_diff_money(r['bt_daily_pnl'])}</td>"
                    f"<td class='{pd_cls}'>{fmt_diff_money(r['pnl_diff'])}</td>"
                    f"<td>{fmt_pct(r['live_daily_return_pct'], sign=True)}</td>"
                    f"<td>{fmt_pct(r['bt_daily_return_pct'], sign=True)}</td>"
                    f"<td class='{pd_cls}'>{fmt_pct(r['ret_diff'], sign=True)}</td>"
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
  <div class='metric'><div class='label'>实盘总资产</div><div class='value'>{fmt_money(s['live_total_asset'])}</div></div>
  <div class='metric'><div class='label'>回测总资产</div><div class='value'>{fmt_money(s['bt_total_asset'])}</div></div>
  <div class='metric'><div class='label'>实盘日 P&L</div><div class='value'>{fmt_diff_money(s['live_daily_pnl'])} ({fmt_pct(s['live_daily_return_pct'], sign=True)})</div></div>
  <div class='metric'><div class='label'>回测日 P&L</div><div class='value'>{fmt_diff_money(s['bt_daily_pnl'])} ({fmt_pct(s['bt_daily_return_pct'], sign=True)})</div></div>
</div>
{self._reconcile_html(data.get('reconcile') or {})}

<div class='section'>
  <h2>操作差距（多退少补：实盘 vs 回测）</h2>
  <div>买入: 实盘 {fmt_money(d3['live_buy_total'])} · 回测 {fmt_money(d3['bt_buy_total'])} · 买量缺口合计 {d3.get('buy_shares_gap_total', 0):+d} 股</div>
  <div>卖出: 实盘 {fmt_money(d3['live_sell_total'])} · 回测 {fmt_money(d3['bt_sell_total'])}</div>
  {_orders_table(d3['rows'])}
</div>

<div class='section'>
  <h2>滑点明细（成交价 vs 开盘价，仅 debug 用）</h2>
  <div>平均滑点 {d4['avg_slippage']:+.3f}% ({d4['avg_slippage']*100:+.1f} bp) · 总滑点成本 {fmt_diff_money(d4['total_slippage_cost'])}</div>
  {_slippage_table(d4['rows'])}
</div>

<div class='section'>
  <h2>逐股盈亏对比</h2>
  <div>实盘合计 {fmt_diff_money(d5['live_total_pnl'])} · 回测合计 {fmt_diff_money(d5['bt_total_pnl'])}</div>
  {_pnl_table(d5['rows'])}
</div>

</body></html>"""
