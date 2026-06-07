"""飞书卡片共享格式化函数。report / day_board / replay 共用。"""

import pandas as pd


def fmt_money(v: float | None) -> str:
    if v is None or pd.isna(v):
        return '-'
    if abs(v) >= 1e8:
        return f"¥{v/1e8:.2f}亿"
    if abs(v) >= 1e4:
        return f"¥{v/1e4:.1f}w"
    return f"¥{v:,.0f}"


def fmt_pct(v: float | None, sign: bool = False) -> str:
    if v is None or pd.isna(v):
        return '-'
    return f"{v:+.2f}%" if sign else f"{v:.2f}%"


def fmt_diff_money(v: float | None) -> str:
    if v is None or pd.isna(v):
        return '-'
    if v == 0:
        return '0'
    if abs(v) >= 1e4:
        return f"{v/1e4:+.1f}w"
    return f"{v:+,.0f}"


def diff_md(v: float | None) -> str:
    """正红负绿（中国股票配色），用于 diff 列。"""
    if v is None or pd.isna(v):
        return '-'
    s = fmt_diff_money(v)
    if v > 0:
        return f'<font color="red">{s}</font>'
    if v < 0:
        return f'<font color="green">{s}</font>'
    return s


def diff_pct_md(v: float | None) -> str:
    if v is None or pd.isna(v):
        return '-'
    s = fmt_pct(v, sign=True)
    if v > 0:
        return f'<font color="red">{s}</font>'
    if v < 0:
        return f'<font color="green">{s}</font>'
    return s


def fmt_shares(n: int) -> str:
    return f"{int(n):,}"


def fmt_price_qty(price: float, vol: int) -> str:
    if vol == 0 and (not price or price == 0):
        return '0'
    return f"{price:.2f} × {fmt_shares(vol)}"


def gap_md(plan_amt: float, done_amt: float) -> str:
    """「计划→成交」缺口: > ¥1000 标红, 否则打满打 ✓。"""
    seg = f"{fmt_money(plan_amt)}→{fmt_money(done_amt)}"
    gap = plan_amt - done_amt
    if gap > 1000:
        return f'{seg} <font color="red">缺{fmt_money(gap)}</font>'
    return f'{seg} ✓'


def diff_shares_md(v: int) -> str:
    """持仓股数 diff：0=对齐(绿)，非0=尚未对齐(橙)。"""
    if v == 0:
        return '<font color="green">0</font>'
    return f'<font color="orange">{v:+,}股</font>'
