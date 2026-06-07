"""多日回放报告 — 遍历历史 parquet → 逐日飞书日报。

入口: `replay_reports(start_date, end_date, individual_config, factor_classes)`
和实盘走同一套逻辑：每张日报用 T-1 种子回放算出回测收益，卡片与实盘完全一致。
"""
from __future__ import annotations
from datetime import date
from pathlib import Path

import pandas as pd

from trading.lark.format import fmt_pct, fmt_diff_money, fmt_shares
from trading.lark.sender import lark_sender, LarkMsgLevel, make_v2_card, make_v2_table, md_div
from trading.logger import trading_logger
from utils.stock.time import get_trading_date_span

_TRADE_DIR = Path(__file__).resolve().parents[1] / "data" / "live_trades"


def _read_plan(trade_date: date) -> pd.DataFrame | None:
    p = _TRADE_DIR / f"plan_{trade_date.isoformat()}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _read_fills(trade_date: date) -> pd.DataFrame | None:
    p = _TRADE_DIR / f"fills_{trade_date.isoformat()}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _read_summary_row(trade_date: date) -> dict | None:
    p = _TRADE_DIR / "daily_summary.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    rows = df[df['date'] == trade_date]
    return rows.iloc[-1].to_dict() if not rows.empty else None


def _read_positions(trade_date: date) -> pd.DataFrame | None:
    p = _TRADE_DIR / f"positions_{trade_date.isoformat()}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _build_daily_card(trade_date: date, plan_df: pd.DataFrame | None,
                      fills_df: pd.DataFrame | None,
                      pos_df: pd.DataFrame | None,
                      summary: dict | None,
                      bt_daily_return: float | None = None) -> dict:
    """单张统一表：5 列 — 股票|持仓手数|操作|diff差异|实盘日志。日收益在标题。"""
    pnl = summary.get('daily_pnl') if summary else None
    ret = summary.get('daily_return_pct') if summary else None
    level = LarkMsgLevel.Info
    sub_parts = []
    if pnl is not None:
        sub_parts.append(f"实盘盈亏 {fmt_diff_money(pnl)}")
    if ret is not None:
        sub_parts.append(f"实盘 {fmt_pct(ret, sign=True)}")
        level = LarkMsgLevel.Success if ret >= 0 else LarkMsgLevel.Danger
    if bt_daily_return is not None:
        sub_parts.append(f"回测 {fmt_pct(bt_daily_return, sign=True)}")
    sub = ' · '.join(sub_parts) if sub_parts else '无汇总数据'

    name_map: dict[str, str] = {}
    pos_map: dict[str, int] = {}
    if pos_df is not None and not pos_df.empty:
        for _, r in pos_df.iterrows():
            pos_map[r['code']] = int(r['volume'])
            if r.get('name'):
                name_map[r['code']] = str(r['name'])

    plan_buy: dict[str, dict] = {}
    plan_sell: dict[str, dict] = {}
    if plan_df is not None:
        for _, r in plan_df.iterrows():
            nm = r.get('name')
            if nm and isinstance(nm, str):
                name_map[r['code']] = nm
            d = {'est_volume': int(r.get('est_volume', 0) or 0),
                 'est_amount': float(r.get('est_amount', 0) or 0),
                 'limit_status': r.get('limit_status', 'ok'),
                 'reason': r.get('reason', '') or ''}
            if r['direction'] == 'buy':
                plan_buy[r['code']] = d
            else:
                plan_sell[r['code']] = d

    fill_map: dict[str, dict] = {}
    if fills_df is not None and not fills_df.empty:
        for code, grp in fills_df.groupby('code'):
            buys = grp[grp['direction'] == 'buy']
            sells = grp[grp['direction'] == 'sell']
            fill_map[code] = {
                'buy_shares': int(buys['shares'].sum()),
                'buy_amt': float(buys['amount'].sum()),
                'sell_shares': int(sells['shares'].sum()),
                'sell_amt': float(sells['amount'].sum()),
            }
            nm = grp['name'].iloc[0] if 'name' in grp.columns and not grp.empty else None
            if nm and isinstance(nm, str):
                name_map[code] = nm

    def _diff_text(tgt_sh: int, act_sh: int) -> str:
        d = act_sh - tgt_sh
        if d == 0:
            return '<font color="grey">—</font>'
        if tgt_sh >= 0:
            word = '多买' if d > 0 else '少买'
        else:
            word = '少卖' if d > 0 else '多卖'
        color = 'red' if d > 0 else 'green'
        return f'<font color="{color}">实盘{word}{abs(d)}股</font>'

    from data.db.stock_name import get_stock_name_at_date
    for code in set(pos_map) | set(plan_buy) | set(plan_sell) | set(fill_map):
        if code not in name_map:
            n = get_stock_name_at_date(code, trade_date)
            if n:
                name_map[code] = n

    all_codes = set(pos_map) | set(plan_buy) | set(plan_sell) | set(fill_map)

    table = []
    for code in sorted(all_codes):
        name = name_map.get(code, code)

        pb = plan_buy.get(code, {})
        ps = plan_sell.get(code, {})
        fm = fill_map.get(code, {})
        vol = pos_map.get(code, 0)
        plan_net_sh = pb.get('est_volume', 0) - ps.get('est_volume', 0)
        fill_net_sh = fm.get('buy_shares', 0) - fm.get('sell_shares', 0)
        target_vol = vol - fill_net_sh + plan_net_sh
        if target_vol == vol and vol > 0:
            pos_cell = f'<font color="grey">目标{fmt_shares(vol)} vs 实盘{fmt_shares(vol)}</font>'
        elif target_vol or vol:
            pos_cell = f'目标{fmt_shares(target_vol)} vs 实盘{fmt_shares(vol)}'
        else:
            pos_cell = '-'

        if plan_net_sh or fill_net_sh:
            ops_cell = f'目标{plan_net_sh:+d} vs 实盘{fill_net_sh:+d}'
        else:
            ops_cell = '-'

        diff_cell = _diff_text(plan_net_sh, fill_net_sh) if (plan_net_sh or fill_net_sh) else '-'

        log_parts = []
        if pb and pb.get('est_volume', 0) > 0:
            f_sh = fm.get('buy_shares', 0)
            if f_sh >= pb['est_volume']:
                log_parts.append('<font color="green">✅ 买入已成</font>')
            elif f_sh > 0:
                log_parts.append(f'<font color="orange">⚠ 买入部成 {f_sh}/{pb["est_volume"]}</font>')
            elif pb.get('limit_status') != 'ok':
                log_parts.append(f'<font color="red">{pb.get("reason","未下单")}</font>')
            else:
                log_parts.append('<font color="red">❌ 未成交</font>')
        if ps and ps.get('est_volume', 0) > 0:
            f_sh = fm.get('sell_shares', 0)
            if f_sh >= ps['est_volume']:
                log_parts.append('<font color="green">✅ 卖出已成</font>')
            elif f_sh > 0:
                log_parts.append(f'<font color="orange">⚠ 卖出部成 {f_sh}/{ps["est_volume"]}</font>')
            else:
                log_parts.append('<font color="red">❌ 未成交</font>')
        if not log_parts and vol > 0:
            log_parts.append('<font color="grey">—</font>')
        log_cell = '\n'.join(log_parts) if log_parts else '-'

        table.append({'name': name, 'pos': pos_cell, 'ops': ops_cell,
                      'diff': diff_cell, 'log': log_cell})

    total_vol = sum(pos_map.values())
    total_plan_sh = sum(pb.get('est_volume', 0) for pb in plan_buy.values()) - sum(ps.get('est_volume', 0) for ps in plan_sell.values())
    total_fill_sh = sum(fm.get('buy_shares', 0) for fm in fill_map.values()) - sum(fm.get('sell_shares', 0) for fm in fill_map.values())
    total_target_vol = total_vol - total_fill_sh + total_plan_sh
    table.append({
        'name': '— 合计 —',
        'pos': f'目标{fmt_shares(total_target_vol)} vs 实盘{fmt_shares(total_vol)}',
        'ops': f'目标{total_plan_sh:+d} vs 实盘{total_fill_sh:+d}',
        'diff': _diff_text(total_plan_sh, total_fill_sh),
        'log': f'<font color="grey">{len(table)} 只</font>',
    })

    elements = [make_v2_table(
        columns=[
            {'name': 'name', 'display_name': '股票', 'horizontal_align': 'left'},
            {'name': 'pos', 'display_name': '持仓手数(目标vs实盘)', 'horizontal_align': 'right'},
            {'name': 'ops', 'display_name': '操作(目标vs实盘)', 'horizontal_align': 'right'},
            {'name': 'diff', 'display_name': 'diff差异', 'horizontal_align': 'right'},
            {'name': 'log', 'display_name': '实盘日志', 'horizontal_align': 'left'},
        ],
        rows=table, element_id='daily_tbl', page_size=20)]

    elements.append({'tag': 'hr'})
    elements.append(md_div(f'<font color="grey">日报 {trade_date.isoformat()}</font>'))

    return make_v2_card(
        title=f"实盘日报 @ {trade_date.isoformat()}",
        level=level, subtitle=sub, elements=elements)


def replay_reports(start_date: date, end_date: date | None = None,
                   individual_config: dict | None = None,
                   factor_classes: list | None = None):
    """逐日回放 → 飞书日报。与实盘共用 _run_seed_replay 算回测收益。"""
    end = end_date or date.today()
    days = get_trading_date_span(start_date, end)
    if not days:
        trading_logger.warning(f"[Replay] {start_date} → {end} 无交易日")
        return

    from trading.post_close import _run_seed_replay

    trading_logger.info(f"[Replay] {len(days)} 个交易日: {days[0].isoformat()} → {days[-1].isoformat()}")
    sent = 0
    for td in days:
        plan = _read_plan(td)
        fills = _read_fills(td)
        summary = _read_summary_row(td)
        positions = _read_positions(td)
        if plan is None and fills is None and summary is None and positions is None:
            continue

        bt_daily_return = None
        if individual_config and factor_classes:
            bt_result = _run_seed_replay(td, individual_config, factor_classes)
            if bt_result is not None:
                snaps = bt_result.get('daily_snapshots') or []
                if snaps:
                    bt_daily_return = snaps[-1].get('daily_return_pct')

        try:
            card = _build_daily_card(td, plan, fills, positions, summary,
                                     bt_daily_return=bt_daily_return)
            lark_sender.send_card(card)
            sent += 1
            trading_logger.info(f"[Replay] {td.isoformat()} 日报已发送")
        except Exception as e:
            trading_logger.warning(f"[Replay] {td.isoformat()} 日报发送失败: {e}")

    trading_logger.info(f"[Replay] 完成: {sent}/{len(days)} 天已发送")
