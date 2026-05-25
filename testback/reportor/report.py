"""
WBR 回测可视化报告生成器

包含：
1. 策略概览（配置参数）
2. 核心指标（收益率、年化、夏普、最大回撤、卡玛等）
3. 净值曲线（vs 沪深300基准）
4. 每日资金快照表
5. 月度收益统计
6. 交易记录明细表（每笔买/卖）
7. 胜率/盈亏分析
8. 当前持仓明细
9. 每日收益率分布
"""

from datetime import datetime, date as date_type, timedelta
from html import escape as html_escape
from pathlib import Path
import re
from typing import Any, Dict, List

import numpy as np

from testback.logger import testback_logger
from testback.metrics import (
    compute_hs300_cumulative_returns,
    compute_strategy_metrics,
)


# ---------------------------------------------------------------------------
# 通用格式化
# ---------------------------------------------------------------------------


def _normalize_date(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if hasattr(value, 'strftime'):
        return value.strftime('%Y-%m-%d')
    return str(value)



def _fmt_money(v: float) -> str:
    return f'¥{v:,.2f}'



def _fmt_pct(v: float, sign: bool = True) -> str:
    prefix = '+' if sign and v > 0 else ''
    return f'{prefix}{v:.2f}%'



def _get_stock_name(code: str, stock_name_map: Dict[str, str]) -> str:
    if not code:
        return ''
    return (stock_name_map or {}).get(code, '') or ''



def _fmt_stock(code: str, stock_name_map: Dict[str, str]) -> str:
    if not code:
        return ''
    name = _get_stock_name(code, stock_name_map)
    return f'{code} {name}' if name else code


def _minify_html_document(html: str) -> str:
    """轻量级 HTML 压缩，避免破坏内联 JS/CSS。"""
    blocks = re.split(r'(<script\b.*?</script>|<style\b.*?</style>)', html, flags=re.IGNORECASE | re.DOTALL)
    minified: List[str] = []
    for block in blocks:
        if not block:
            continue
        if block.lstrip().lower().startswith('<script') or block.lstrip().lower().startswith('<style'):
            minified.append(block.strip())
            continue
        compact = re.sub(r'<!--.*?-->', '', block, flags=re.DOTALL)
        compact = re.sub(r'>\s+<', '><', compact)
        compact = re.sub(r'\n\s+', '\n', compact)
        minified.append(compact.strip())
    return ''.join(minified)



def _make_stock_text(code: str, stock_name_map: Dict[str, str]) -> str:
    return f'<span class="stock-text">{html_escape(_fmt_stock(code, stock_name_map))}</span>'



def _make_stock_button(code: str, stock_name_map: Dict[str, str], class_name: str = 'code-btn') -> str:
    label = html_escape(_fmt_stock(code, stock_name_map))
    return f'<span class="{class_name}">{label}</span>'



def _make_stock_list_html(codes: List[str], stock_name_map: Dict[str, str]) -> str:
    if not codes:
        return '<span class="muted">—</span>'
    return '<br>'.join(html_escape(_fmt_stock(code, stock_name_map)) for code in codes)



def _make_stock_action_list_html(buys: List[str], sells: List[str], stock_name_map: Dict[str, str]) -> str:
    parts = []
    if buys:
        buy_html = '<br>'.join(
            f'<span class="buy-cell">买 {html_escape(_fmt_stock(code, stock_name_map))}</span>'
            for code in buys
        )
        parts.append(buy_html)
    if sells:
        sell_html = '<br>'.join(
            f'<span class="sell-cell">卖 {html_escape(_fmt_stock(code, stock_name_map))}</span>'
            for code in sells
        )
        parts.append(sell_html)
    if not parts:
        return '<span class="muted">—</span>'
    return '<br>'.join(parts)



def _make_stock_count_summary(codes: List[str], stock_name_map: Dict[str, str], empty_label: str = '—') -> tuple[str, str]:
    if not codes:
        return f'<span class="muted">{html_escape(empty_label)}</span>', ''
    title = '\n'.join(_fmt_stock(code, stock_name_map) for code in codes)
    return f'<span>{len(codes)} 只</span>{_make_detail_icon(title)}', title


def _tooltip_html(detail: str) -> str:
    if not detail:
        return ''
    return html_escape(detail).replace('\n', '<br>')


def _make_help_label(label: str, tooltip: str) -> str:
    icon_html = _make_detail_icon(tooltip)
    return f'{html_escape(label)}{icon_html}'


def _make_cell(html: str, sort_value: Any = None, cell_class: str = '', title: str | None = None) -> Dict[str, Any]:
    return {
        'html': html,
        'sort': '' if sort_value is None else str(sort_value),
        'class': cell_class,
        'title': title,
    }



def _get_signal_date(record: Dict[str, Any]) -> str:
    return _normalize_date(
        record.get('signal_date')
        or record.get('buy_signal_date')
        or record.get('clear_signal_date')
        or record.get('buy_date')
        or record.get('clear_date')
        or record.get('date')
    )



def _get_trade_date(record: Dict[str, Any]) -> str:
    return _normalize_date(
        record.get('trade_date')
        or record.get('buy_trade_date')
        or record.get('clear_trade_date')
        or record.get('buy_date')
        or record.get('clear_date')
        or record.get('date')
    )



def _format_execution_basis(record: Dict[str, Any]) -> str:
    price_field = record.get('price_field') or 'close'
    signal_dividend_type = record.get('signal_dividend_type') or 'back'
    execution_dividend_type = record.get('execution_dividend_type') or signal_dividend_type
    return f'{price_field} | {signal_dividend_type}->{execution_dividend_type}'


# ---------------------------------------------------------------------------
# K线图生成
# ---------------------------------------------------------------------------


def _get_stock_trades(trade_log: List[Dict], code: str) -> Dict:
    """获取某股票的所有买卖点"""
    trades = [t for t in trade_log if t.get('code') == code]
    buys = [
        {
            'signal_date': _get_signal_date(t),
            'trade_date': _get_trade_date(t),
            'price': t.get('price', 0),
            'volume': t.get('volume', 0),
            'execution_basis': _format_execution_basis(t),
        }
        for t in trades if t.get('action') == 'buy'
    ]
    sells = [
        {
            'signal_date': _get_signal_date(t),
            'trade_date': _get_trade_date(t),
            'price': t.get('price', 0),
            'volume': t.get('volume', 0),
            'income': t.get('income'),
            'execution_basis': _format_execution_basis(t),
        }
        for t in trades if t.get('action') == 'sell'
    ]
    return {'buys': buys, 'sells': sells}



def _build_kline_chart(code: str, trade_log: List[Dict], trade_dates: List[str],
                       stock_name_map: Dict[str, str]) -> str:
    """为单只股票生成K线图（蜡烛图+成交量+买卖点标记）"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from data.db import get_market_data_from_cache
    except Exception:
        return ''

    if not trade_dates:
        return ''

    try:
        from datetime import timedelta
        first_date = datetime.strptime(trade_dates[0], '%Y-%m-%d')
        last_date = datetime.strptime(trade_dates[-1], '%Y-%m-%d')
        start_dt = first_date - timedelta(days=60)
        end_dt = last_date + timedelta(days=60)
    except Exception:
        return ''

    try:
        data = get_market_data_from_cache(code, 300, end_dt, '1d', dividend_type='back')
    except Exception:
        return ''

    if data is None or len(data.get('close', [])) < 5:
        return ''

    timestamps = data['time']
    dates = [datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d') for ts in timestamps]
    opens = data['open']
    highs = data['high']
    lows = data['low']
    closes = data['close']
    volumes = data['amount']

    stock_trades = _get_stock_trades(trade_log, code)
    buys = stock_trades['buys']
    sells = stock_trades['sells']

    buy_x, buy_y = [], []
    for b in buys:
        d_str = b['trade_date']
        if d_str in dates:
            idx = dates.index(d_str)
            buy_x.append(d_str)
            buy_y.append(float(b.get('price', closes[idx])))

    sell_x, sell_y = [], []
    for s in sells:
        d_str = s['trade_date']
        if d_str in dates:
            idx = dates.index(d_str)
            sell_x.append(d_str)
            sell_y.append(float(s.get('price', closes[idx])))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.72, 0.28],
        subplot_titles=('', '成交量'),
    )

    fig.add_trace(go.Candlestick(
        x=dates,
        open=opens,
        high=highs,
        low=lows,
        close=closes,
        name='K线',
        increasing_line_color='#26A69A',
        decreasing_line_color='#EF5350',
        increasing_fillcolor='#26A69A',
        decreasing_fillcolor='#EF5350',
        opacity=0.9,
    ), row=1, col=1)

    if buy_x:
        fig.add_trace(go.Scatter(
            x=buy_x,
            y=buy_y,
            mode='markers',
            name='买入',
            marker=dict(symbol='triangle-up', size=14, color='#2E7D32', line=dict(width=1, color='white')),
            hovertemplate='买入<br>执行日: %{x}<br>价格: %{y:.4f}<extra></extra>',
        ), row=1, col=1)

    if sell_x:
        fig.add_trace(go.Scatter(
            x=sell_x,
            y=sell_y,
            mode='markers',
            name='卖出',
            marker=dict(symbol='triangle-down', size=14, color='#C62828', line=dict(width=1, color='white')),
            hovertemplate='卖出<br>执行日: %{x}<br>价格: %{y:.4f}<extra></extra>',
        ), row=1, col=1)

    colors = ['#26A69A' if c >= o else '#EF5350' for c, o in zip(closes, opens)]
    fig.add_trace(go.Bar(
        x=dates,
        y=volumes,
        name='成交量',
        marker_color=colors,
        opacity=0.7,
        hovertemplate='<b>%{x}</b><br>成交额: %{y:,.0f}<extra></extra>',
    ), row=2, col=1)

    display_code = _fmt_stock(code, stock_name_map)
    fig.update_layout(
        title=dict(text=f'K线走势 · {display_code}', x=0.5, font=dict(size=15)),
        height=500,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(title_text='', row=2, col=1)
    fig.update_yaxes(title_text='价格', row=1, col=1)
    fig.update_yaxes(title_text='成交额', row=2, col=1)

    return fig.to_html(full_html=False, include_plotlyjs='cdn')



# ---------------------------------------------------------------------------
# 基准数据
# ---------------------------------------------------------------------------




def _cumulative_returns_to_nav(cumulative_returns_pct: List[float]) -> List[float]:
    return [round(1.0 + float(value) / 100.0, 6) for value in cumulative_returns_pct]



def _cumulative_to_daily_returns(cumulative_returns_pct: List[float]) -> List[float]:
    if not cumulative_returns_pct:
        return []

    nav = np.array(_cumulative_returns_to_nav(cumulative_returns_pct), dtype=float)
    prev_nav = np.concatenate(([1.0], nav[:-1]))
    valid_prev_nav = np.where(prev_nav == 0, np.nan, prev_nav)
    daily = (nav / valid_prev_nav - 1.0) * 100.0
    daily = np.nan_to_num(daily, nan=0.0, posinf=0.0, neginf=0.0)
    return [round(float(value), 6) for value in daily.tolist()]



def _resolve_daily_returns_pct(report_data: Dict[str, Any],
                               cumulative_returns_pct: List[float],
                               trade_dates: List[str]) -> List[float]:
    daily_snapshots = report_data.get('daily_snapshots') or []
    if daily_snapshots and len(daily_snapshots) == len(trade_dates):
        return [round(float(snapshot.get('daily_return_pct', 0.0)), 6) for snapshot in daily_snapshots]

    stored_daily_returns = report_data.get('daily_returns') or []
    if stored_daily_returns and len(stored_daily_returns) == len(trade_dates):
        same_as_cumulative = all(
            abs(float(stored_daily_returns[idx]) - float(cumulative_returns_pct[idx])) < 1e-9
            for idx in range(len(trade_dates))
        )
        if not same_as_cumulative:
            return [round(float(value), 6) for value in stored_daily_returns]

    return _cumulative_to_daily_returns(cumulative_returns_pct)



def calc_monthly_stats(trade_dates: List[str],
                       cumulative_returns_pct: List[float],
                       daily_returns_pct: List[float] | None = None) -> List[Dict]:
    """计算月度收益统计。"""
    if not trade_dates or not cumulative_returns_pct:
        return []

    daily_returns_pct = daily_returns_pct or _cumulative_to_daily_returns(cumulative_returns_pct)
    nav_values = _cumulative_returns_to_nav(cumulative_returns_pct)

    monthly: Dict[str, Dict[str, Any]] = {}
    prev_month_end_nav = 1.0

    for idx, trade_date in enumerate(trade_dates):
        month = trade_date[:7]
        if month not in monthly:
            monthly[month] = {
                'start_nav': prev_month_end_nav,
                'end_nav': prev_month_end_nav,
                'trade_days': 0,
                'win_days': 0,
                'lose_days': 0,
            }

        bucket = monthly[month]
        bucket['trade_days'] += 1
        bucket['end_nav'] = nav_values[idx]
        if daily_returns_pct[idx] > 0:
            bucket['win_days'] += 1
        elif daily_returns_pct[idx] < 0:
            bucket['lose_days'] += 1
        prev_month_end_nav = nav_values[idx]

    result = []
    for month in sorted(monthly):
        item = monthly[month]
        start_nav = item['start_nav']
        end_nav = item['end_nav']
        monthly_return = (end_nav / start_nav - 1.0) * 100.0 if start_nav else 0.0
        cumulative_return = (end_nav - 1.0) * 100.0
        trade_days = item['trade_days']
        result.append({
            'month': month,
            'monthly_return': round(monthly_return, 2),
            'cumulative_return': round(cumulative_return, 2),
            'trade_days': trade_days,
            'win_days': item['win_days'],
            'lose_days': item['lose_days'],
            'win_rate': round(item['win_days'] / trade_days * 100.0, 1) if trade_days else 0.0,
        })

    return result



def _build_histogram_payload(values: List[float], max_bins: int = 32) -> Dict[str, Any]:
    if not values or len(values) < 2:
        return {}

    arr = np.array(values, dtype=float)
    bin_count = min(max_bins, max(10, int(np.sqrt(len(arr)))))
    counts, edges = np.histogram(arr, bins=bin_count)
    labels = [
        f'{edges[idx]:.2f}% ~ {edges[idx + 1]:.2f}%'
        for idx in range(len(edges) - 1)
    ]
    return {
        'labels': labels,
        'counts': counts.astype(int).tolist(),
        'mean': round(float(np.mean(arr)), 4),
        'std': round(float(np.std(arr, ddof=1)), 4) if len(arr) > 1 else 0.0,
    }



def _build_winloss_payload(trade_log: List[Dict]) -> Dict[str, int]:
    sell_trades = [trade for trade in trade_log if trade.get('action') == 'sell' and trade.get('income') is not None]
    if not sell_trades:
        return {}

    wins = len([trade for trade in sell_trades if trade.get('income', 0) > 0])
    losses = len(sell_trades) - wins
    return {'wins': wins, 'losses': losses}


# ---------------------------------------------------------------------------
# 表格生成
# ---------------------------------------------------------------------------


def _make_table(headers: List[Any], rows: List[List[Any]],
                id_attr: str = '',
                empty_message: str = '',
                row_height: int = 44,
                max_height: int = 460) -> Dict[str, Any]:
    normalized_headers = []
    for idx, header in enumerate(headers):
        if isinstance(header, dict):
            label = header.get('label', '')
            sort_type = header.get('sort_type', 'text')
            sortable = header.get('sortable', True)
            align = header.get('align') or (
                'right' if sort_type == 'number' else
                'center' if sort_type == 'date' else
                'left'
            )
        else:
            label = str(header)
            sort_type = 'text'
            sortable = True
            align = 'left'
        normalized_headers.append({
            'id': f'c{idx}',
            'label': label,
            'sort_type': sort_type,
            'sortable': sortable,
            'align': align,
            'size': header.get('size') if isinstance(header, dict) and header.get('size') is not None else (
                128 if sort_type == 'date' else
                120 if sort_type == 'number' else
                280 if '列表' in label or '买卖' in label or '原因' in label or '说明' in label else
                180
            ),
        })

    normalized_rows = []
    for row in rows:
        normalized_row = []
        for cell in row:
            if isinstance(cell, dict):
                cell_html = cell.get('html', '')
                sort_value = '' if cell.get('sort') is None else str(cell.get('sort', ''))
                cell_class = cell.get('class', '')
                title = str(cell.get('title', sort_value))
            else:
                cell_html = str(cell)
                sort_value = str(cell)
                cell_class = ''
                title = str(cell)
            normalized_row.append({
                'html': cell_html,
                'sort': sort_value,
                'class': cell_class,
                'title': title,
            })
        normalized_rows.append(normalized_row)

    return {
        'table_id': id_attr,
        'headers': normalized_headers,
        'rows': normalized_rows,
        'empty_message': empty_message,
        'row_height': row_height,
        'max_height': max_height,
    }



def _make_detail_icon(detail: str) -> str:
    if not detail or not detail.strip() or detail.strip().lower() == 'none':
        return ''
    return (
        f'<span class="detail-icon has-tooltip" '
        f'data-tippy-content="{_tooltip_html(detail)}" '
        f'aria-label="详情">'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-question-circle-fill" viewBox="0 0 16 16">'
        f'<path d="M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0M5.496 6.033h.825c.138 0 .248-.113.266-.25.09-.656.54-1.134 1.342-1.134.686 0 1.314.343 1.314 1.168 0 .635-.374.927-.965 1.371-.673.489-1.206 1.06-1.168 1.987l.003.217a.25.25 0 0 0 .25.246h.811a.25.25 0 0 0 .25-.25v-.105c0-.718.273-.927 1.01-1.486.609-.463 1.244-.977 1.244-2.056 0-1.511-1.276-2.241-2.673-2.241-1.267 0-2.655.59-2.75 2.286a.237.237 0 0 0 .241.247m2.325 6.443c.61 0 1.029-.394 1.029-.927 0-.552-.42-.94-1.029-.94-.584 0-1.009.388-1.009.94 0 .533.425.927 1.01.927z"/>'
        f'</svg>'
        f'</span>'
    )


def _make_trade_count_summary(buys: List[str], sells: List[str], stock_name_map: Dict[str, str]) -> tuple[str, str]:
    parts = []
    title_lines = []
    if buys:
        buy_detail = '\n'.join([_fmt_stock(code, stock_name_map) for code in buys])
        parts.append(f'买入 {len(buys)} 只{_make_detail_icon(buy_detail)}')
        title_lines.extend([_fmt_stock(code, stock_name_map) for code in buys])
    if sells:
        sell_detail = '\n'.join([_fmt_stock(code, stock_name_map) for code in sells])
        parts.append(f'卖出 {len(sells)} 只{_make_detail_icon(sell_detail)}')
        title_lines.extend([_fmt_stock(code, stock_name_map) for code in sells])
    if not parts:
        return '<span class="muted">—</span>', ''
    return '<br>'.join(parts), '\n'.join(title_lines)


def _make_daily_table(daily_snapshots: List[Dict], stock_name_map: Dict[str, str]) -> Dict[str, Any]:
    headers = [
        {'label': '信号日', 'sort_type': 'date', 'align': 'center'},
        {'label': '执行日', 'sort_type': 'date', 'align': 'center'},
        {'label': '执行基准', 'sort_type': 'text', 'align': 'center'},
        {'label': '现金(¥)', 'sort_type': 'number'},
        {'label': '持仓市值(¥)', 'sort_type': 'number'},
        {'label': '总资产(¥)', 'sort_type': 'number'},
        {'label': '日收益率', 'sort_type': 'number'},
        {'label': '累计收益率', 'sort_type': 'number'},
        {'label': '仓位', 'sort_type': 'number'},
        {'label': '候选 buy_n 列表', 'sort_type': 'text'},
        {'label': '实际买卖', 'sort_type': 'text'},
    ]
    if not daily_snapshots:
        return _make_table(headers, [], 'daily-table', '暂无每日快照数据', row_height=92, max_height=540)

    rows = []
    for s in daily_snapshots:
        daily_pct = s.get('daily_return_pct', 0)
        cum_pct = s.get('cumulative_return_pct', 0)
        daily_cls = 'pos-cell' if daily_pct > 0 else ('neg-cell' if daily_pct < 0 else '')
        cum_cls = 'pos-cell' if cum_pct > 0 else ('neg-cell' if cum_pct < 0 else '')
        buy_n_list = s.get('buy_n_list', [])
        executed_buy_list = s.get('executed_buy_list', [])
        executed_sell_list = s.get('executed_sell_list', [])
        signal_date = _get_signal_date(s)
        trade_date = _get_trade_date(s)
        execution_basis = _format_execution_basis(s)
        action_sort = '|'.join(
            [f'B:{_fmt_stock(code, stock_name_map)}' for code in executed_buy_list] +
            [f'S:{_fmt_stock(code, stock_name_map)}' for code in executed_sell_list]
        )
        buy_summary_html, buy_list_text = _make_stock_count_summary(buy_n_list, stock_name_map)
        trade_summary_html, trade_summary_title = _make_trade_count_summary(executed_buy_list, executed_sell_list, stock_name_map)
        total_asset = s.get('total_asset', 0)
        market_value = s.get('market_value', 0)
        pos_ratio = (market_value / total_asset * 100) if total_asset > 0 else 0.0
        rows.append([
            _make_cell(html_escape(signal_date), signal_date),
            _make_cell(html_escape(trade_date), trade_date),
            _make_cell(html_escape(execution_basis), execution_basis),
            _make_cell(f'{s["cash"]:,.2f}', s['cash']),
            _make_cell(f'{s["market_value"]:,.2f}', s['market_value']),
            _make_cell(f'{s["total_asset"]:,.2f}', s['total_asset']),
            _make_cell(f'<span class="{daily_cls}">{_fmt_pct(daily_pct)}</span>', daily_pct),
            _make_cell(f'<span class="{cum_cls}">{_fmt_pct(cum_pct)}</span>', cum_pct),
            _make_cell(f'{pos_ratio:.1f}%', pos_ratio),
            _make_cell(buy_summary_html, buy_list_text, title=buy_list_text),
            _make_cell(trade_summary_html, action_sort, title=trade_summary_title),
        ])
    return _make_table(headers, rows, 'daily-table', '暂无每日快照数据', row_height=92, max_height=540)



def _make_trade_table(trade_log: List[Dict], stock_name_map: Dict[str, str]) -> Dict[str, Any]:
    headers = [
        {'label': '信号日', 'sort_type': 'date', 'align': 'center'},
        {'label': '执行日', 'sort_type': 'date', 'align': 'center'},
        {'label': '股票', 'sort_type': 'text'},
        {'label': '方向', 'sort_type': 'text', 'align': 'center'},
        {'label': '执行基准', 'sort_type': 'text', 'align': 'center'},
        {'label': '价格(¥)', 'sort_type': 'number'},
        {'label': '数量(股)', 'sort_type': 'number'},
        {'label': '金额(¥)', 'sort_type': 'number'},
        {'label': '手续费(¥)', 'sort_type': 'number'},
        {'label': '盈亏(¥)', 'sort_type': 'number'},
        {'label': '说明', 'sort_type': 'text'},
    ]
    if not trade_log:
        return _make_table(headers, [], 'trade-table', '暂无交易记录', row_height=60, max_height=540)

    rows = []
    for t in trade_log:
        action = t.get('action', '')
        is_buy = action == 'buy'
        action_cls = 'buy-cell' if is_buy else 'sell-cell'
        action_txt = '买入' if is_buy else '卖出'
        income = t.get('income')
        signal_date = _get_signal_date(t)
        trade_date = _get_trade_date(t)
        execution_basis = _format_execution_basis(t)
        if income is None:
            income_html = '<span class="muted">—</span>'
            income_sort = ''
        else:
            income_cls = 'pos-cell' if income > 0 else 'neg-cell'
            income_html = f'<span class="{income_cls}">{_fmt_money(income)}</span>'
            income_sort = income
        rows.append([
            _make_cell(html_escape(signal_date), signal_date),
            _make_cell(html_escape(trade_date), trade_date),
            _make_cell(_make_stock_button(t.get('code', ''), stock_name_map), _fmt_stock(t.get('code', ''), stock_name_map)),
            _make_cell(f'<span class="{action_cls}">{action_txt}</span>', action_txt),
            _make_cell(html_escape(execution_basis), execution_basis),
            _make_cell(f'{t.get("price", 0):.4f}', t.get('price', 0)),
            _make_cell(f'{t.get("volume", 0):,}', t.get('volume', 0)),
            _make_cell(f'{t.get("amount", 0):,.2f}', t.get('amount', 0)),
            _make_cell(f'{t.get("commission", 0):.2f}', t.get('commission', 0)),
            _make_cell(income_html, income_sort),
            _make_cell(html_escape(t.get('reason', '') or ''), t.get('reason', '') or ''),
        ])
    return _make_table(headers, rows, 'trade-table', '暂无交易记录', row_height=60, max_height=540)



def _make_holdings_table(positions: List[Dict], stock_name_map: Dict[str, str]) -> Dict[str, Any]:
    headers = [
        {'label': '股票', 'sort_type': 'text'},
        {'label': '信号日', 'sort_type': 'date', 'align': 'center'},
        {'label': '执行日', 'sort_type': 'date', 'align': 'center'},
        {'label': '执行基准', 'sort_type': 'text', 'align': 'center'},
        {'label': '持仓天数(交易日)', 'sort_type': 'number'},
        {'label': '持仓数量', 'sort_type': 'number'},
        {'label': '持仓均价(¥)', 'sort_type': 'number'},
        {'label': '持仓成本(¥)', 'sort_type': 'number'},
        {'label': '现价(¥)', 'sort_type': 'number'},
        {'label': '当前市值(¥)', 'sort_type': 'number'},
        {'label': '盈亏(¥)', 'sort_type': 'number'},
        {'label': '盈亏率', 'sort_type': 'number'},
    ]
    if not positions:
        return _make_table(headers, [], 'holdings-table', '当前无持仓', row_height=52, max_height=520)

    rows = []
    for p in positions:
        cur_price = p.get('current_price', 0)
        cur_value = p.get('current_value', 0)
        cost = p.get('cost', 0)
        income = cur_value - cost
        income_pct = (income / cost * 100) if cost > 0 else 0
        income_cls = 'pos-cell' if income > 0 else ('neg-cell' if income < 0 else '')
        signal_date = _get_signal_date(p)
        trade_date = _get_trade_date(p)
        execution_basis = _format_execution_basis(p)
        rows.append([
            _make_cell(_make_stock_button(p.get('code', ''), stock_name_map), _fmt_stock(p.get('code', ''), stock_name_map)),
            _make_cell(html_escape(signal_date), signal_date),
            _make_cell(html_escape(trade_date), trade_date),
            _make_cell(html_escape(execution_basis), execution_basis),
            _make_cell(str(p.get('holding_days', 0)), p.get('holding_days', 0)),
            _make_cell(f'{p.get("volume", 0):,}', p.get('volume', 0)),
            _make_cell(f'{p.get("avg_price", 0):.4f}', p.get('avg_price', 0)),
            _make_cell(f'{cost:,.2f}', cost),
            _make_cell(f'{cur_price:.4f}', cur_price),
            _make_cell(f'{cur_value:,.2f}', cur_value),
            _make_cell(f'<span class="{income_cls}">{_fmt_money(income)}</span>', income),
            _make_cell(f'<span class="{income_cls}">{_fmt_pct(income_pct)}</span>', income_pct),
        ])
    return _make_table(headers, rows, 'holdings-table', '当前无持仓', row_height=52, max_height=520)



def _make_cleared_positions_table(cleared_positions: List[Dict], stock_name_map: Dict[str, str]) -> Dict[str, Any]:
    headers = [
        {'label': '股票', 'sort_type': 'text'},
        {'label': '买入信号日', 'sort_type': 'date', 'align': 'center'},
        {'label': '买入执行日', 'sort_type': 'date', 'align': 'center'},
        {'label': '卖出信号日', 'sort_type': 'date', 'align': 'center'},
        {'label': '卖出执行日', 'sort_type': 'date', 'align': 'center'},
        {'label': '执行基准', 'sort_type': 'text', 'align': 'center'},
        {'label': '持仓天数(交易日)', 'sort_type': 'number'},
        {'label': '数量(股)', 'sort_type': 'number'},
        {'label': '买入均价(¥)', 'sort_type': 'number'},
        {'label': '卖出价格(¥)', 'sort_type': 'number'},
        {'label': '持仓成本(¥)', 'sort_type': 'number'},
        {'label': '盈亏(¥)', 'sort_type': 'number'},
        {'label': '盈亏率', 'sort_type': 'number'},
        {'label': '清仓原因', 'sort_type': 'text'},
    ]
    if not cleared_positions:
        return _make_table(headers, [], 'cleared-table', '暂无已清仓持仓', row_height=60, max_height=520)

    rows = []
    for item in cleared_positions:
        income = item.get('income', 0)
        income_pct = item.get('income_pct', 0)
        income_cls = 'pos-cell' if income > 0 else ('neg-cell' if income < 0 else '')
        buy_signal_date = _normalize_date(item.get('buy_signal_date') or item.get('buy_date', ''))
        buy_trade_date = _normalize_date(item.get('buy_trade_date') or item.get('buy_date', ''))
        clear_signal_date = _normalize_date(item.get('clear_signal_date') or item.get('clear_date', ''))
        clear_trade_date = _normalize_date(item.get('clear_trade_date') or item.get('clear_date', ''))
        execution_basis = _format_execution_basis(item)
        rows.append([
            _make_cell(_make_stock_button(item.get('code', ''), stock_name_map), _fmt_stock(item.get('code', ''), stock_name_map)),
            _make_cell(html_escape(buy_signal_date), buy_signal_date),
            _make_cell(html_escape(buy_trade_date), buy_trade_date),
            _make_cell(html_escape(clear_signal_date), clear_signal_date),
            _make_cell(html_escape(clear_trade_date), clear_trade_date),
            _make_cell(html_escape(execution_basis), execution_basis),
            _make_cell(str(item.get('holding_days', 0)), item.get('holding_days', 0)),
            _make_cell(f'{item.get("volume", 0):,}', item.get('volume', 0)),
            _make_cell(f'{item.get("avg_price", 0):.4f}', item.get('avg_price', 0)),
            _make_cell(f'{item.get("clear_price", 0):.4f}', item.get('clear_price', 0)),
            _make_cell(f'{item.get("cost", 0):,.2f}', item.get('cost', 0)),
            _make_cell(f'<span class="{income_cls}">{_fmt_money(income)}</span>', income),
            _make_cell(f'<span class="{income_cls}">{_fmt_pct(income_pct)}</span>', income_pct),
            _make_cell(html_escape(item.get('clear_reason', '') or ''), item.get('clear_reason', '') or ''),
        ])
    return _make_table(headers, rows, 'cleared-table', '暂无已清仓持仓', row_height=60, max_height=520)



def _make_delist_events_table(delist_events: List[Dict], stock_name_map: Dict[str, str]) -> Dict[str, Any]:
    headers = [
        {'label': '股票', 'sort_type': 'text'},
        {'label': '退市日', 'sort_type': 'date', 'align': 'center'},
        {'label': '归零信号日', 'sort_type': 'date', 'align': 'center'},
        {'label': '归零执行日', 'sort_type': 'date', 'align': 'center'},
        {'label': '买入执行日', 'sort_type': 'date', 'align': 'center'},
        {'label': '持仓天数(交易日)', 'sort_type': 'number'},
        {'label': '数量(股)', 'sort_type': 'number'},
        {'label': '持仓成本(¥)', 'sort_type': 'number'},
        {'label': '归零损失(¥)', 'sort_type': 'number'},
        {'label': '损失率', 'sort_type': 'number'},
        {'label': '说明', 'sort_type': 'text'},
    ]
    if not delist_events:
        return _make_table(headers, [], 'delist-table', '暂无退市归零事件', row_height=56, max_height=520)

    rows = []
    for item in delist_events:
        income = item.get('income', 0)
        income_pct = item.get('income_pct', 0)
        income_cls = 'neg-cell' if income < 0 else ''
        rows.append([
            _make_cell(_make_stock_button(item.get('code', ''), stock_name_map), _fmt_stock(item.get('code', ''), stock_name_map)),
            _make_cell(html_escape(_normalize_date(item.get('delist_date'))), _normalize_date(item.get('delist_date'))),
            _make_cell(html_escape(_normalize_date(item.get('clear_signal_date'))), _normalize_date(item.get('clear_signal_date'))),
            _make_cell(html_escape(_normalize_date(item.get('clear_trade_date'))), _normalize_date(item.get('clear_trade_date'))),
            _make_cell(html_escape(_normalize_date(item.get('buy_trade_date'))), _normalize_date(item.get('buy_trade_date'))),
            _make_cell(str(item.get('holding_days', 0)), item.get('holding_days', 0)),
            _make_cell(f'{item.get("volume", 0):,}', item.get('volume', 0)),
            _make_cell(f'{item.get("cost", 0):,.2f}', item.get('cost', 0)),
            _make_cell(f'<span class="{income_cls}">{_fmt_money(income)}</span>', income),
            _make_cell(f'<span class="{income_cls}">{_fmt_pct(income_pct)}</span>', income_pct),
            _make_cell(html_escape(item.get('clear_reason', '') or ''), item.get('clear_reason', '') or ''),
        ])
    return _make_table(headers, rows, 'delist-table', '暂无退市归零事件', row_height=56, max_height=520)



def _make_monthly_table(monthly_stats: List[Dict]) -> Dict[str, Any]:
    headers = [
        {'label': '月份', 'sort_type': 'text', 'align': 'center'},
        {'label': '月收益率', 'sort_type': 'number'},
        {'label': '累计收益率', 'sort_type': 'number'},
        {'label': '交易天数', 'sort_type': 'number'},
        {'label': '盈利天数', 'sort_type': 'number'},
        {'label': '亏损天数', 'sort_type': 'number'},
        {'label': '日胜率', 'sort_type': 'number'},
    ]
    if not monthly_stats:
        return _make_table(headers, [], 'monthly-table', '暂无月度数据', row_height=44, max_height=420)

    rows = []
    for m in monthly_stats:
        mr = m['monthly_return']
        mr_cls = 'pos-cell' if mr > 0 else ('neg-cell' if mr < 0 else '')
        cum = m['cumulative_return']
        cum_cls = 'pos-cell' if cum > 0 else ('neg-cell' if cum < 0 else '')
        rows.append([
            _make_cell(html_escape(m['month']), m['month']),
            _make_cell(f'<span class="{mr_cls}">{_fmt_pct(mr)}</span>', mr),
            _make_cell(f'<span class="{cum_cls}">{_fmt_pct(cum)}</span>', cum),
            _make_cell(str(m['trade_days']), m['trade_days']),
            _make_cell(str(m['win_days']), m['win_days']),
            _make_cell(str(m['lose_days']), m['lose_days']),
            _make_cell(f"{m['win_rate']:.1f}%", m['win_rate']),
        ])
    return _make_table(headers, rows, 'monthly-table', '暂无月度数据', row_height=44, max_height=420)


# ---------------------------------------------------------------------------
# Plotly 图表生成
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 核心指标卡片 / 统计表
# ---------------------------------------------------------------------------


def _build_per_year_table(per_year_metrics: List[Dict]) -> str:
    if not per_year_metrics:
        return '<p style="color:var(--muted);padding:8px">无年度数据</p>'
    header = '<tr><th>年份</th><th>年化收益</th><th>夏普比率</th><th>最大回撤</th><th>收益</th><th>交易日</th></tr>'
    rows = ''
    for m in per_year_metrics:
        ret_cls = 'pos' if m['return'] >= 0 else 'neg'
        ann_cls = 'pos' if m['annualized'] >= 0 else 'neg'
        shp_cls = 'pos' if m['sharpe'] >= 0 else 'neg'
        rows += (
            f'<tr><td>{m["year"]}</td>'
            f'<td class="{ann_cls}">{m["annualized"]:+.2f}%</td>'
            f'<td class="{shp_cls}">{m["sharpe"]:+.2f}</td>'
            f'<td class="neg">{m["max_drawdown"]:.2f}%</td>'
            f'<td class="{ret_cls}">{m["return"]:+.2f}%</td>'
            f'<td>{m["trading_days"]}</td></tr>'
        )
    return f'<table class="per-year-table"><thead>{header}</thead><tbody>{rows}</tbody></table>'


def _build_metric_cards(metrics: Dict, holding_stats: Dict) -> str:
    max_dd_period = ''
    if metrics.get('max_drawdown_start') and metrics.get('max_drawdown_end'):
        max_dd_period = f"{metrics['max_drawdown_start']} ~ {metrics['max_drawdown_end']}"

    cards = [
        ('总收益率', _fmt_pct(metrics.get('total_return', 0)),
         'pos' if metrics.get('total_return', 0) >= 0 else 'neg',
         '回测结束时总资产相对初始资金的收益率。'),
        ('年化收益率', _fmt_pct(metrics.get('annualized', 0)),
         'pos' if metrics.get('annualized', 0) >= 0 else 'neg',
         '按总收益率和持有时长折算的复利年化收益率。'),
        ('最大回撤', _fmt_pct(metrics.get('max_drawdown', 0), sign=False),
         'neg',
         f'净值从阶段高点回落到阶段低点的最大跌幅。区间: {max_dd_period}' if max_dd_period else '净值从阶段高点回落到阶段低点的最大跌幅。'),
        ('夏普比率', f"{metrics.get('sharpe_ratio', 0):.2f}",
         'neutral',
         '使用日收益率均值和标准差按 252 交易日年化得到的夏普比率。'),
        ('实际买入次数', str(metrics.get('executed_buy_count', 0)),
         'neutral',
         '本次回测实际成交的买入笔数。'),
        ('实际卖出次数', str(metrics.get('executed_sell_count', 0)),
         'neutral',
         '本次回测实际成交的卖出笔数。'),
        ('完整 round-trip', str(metrics.get('round_trip_count', 0)),
         'neutral',
         '已完成买入并完成对应卖出的持仓数量。'),
        ('平均持仓天数', f"{metrics.get('average_holding_days', 0):.2f}",
         'neutral',
         '当前持仓与已清仓持仓合并计算的平均持仓交易日。'),
        ('退市归零次数', str(metrics.get('delist_count', 0)),
         'neutral',
         '持仓股票在退市后被按零价值核销的次数。'),
        ('已清仓持仓数', str(metrics.get('cleared_positions_count', 0)),
         'neutral',
         '报告期内已经完全清仓的持仓数量。'),
        ('当前持仓数', str(metrics.get('current_positions_count', 0)),
         'neutral',
         '回测结束时仍然持有的股票数量。'),
        ('盈利清仓数', str(metrics.get('wins', 0)),
         'neutral',
         '清仓后实现正收益的持仓数量。'),
        ('亏损清仓数', str(metrics.get('losses', 0)),
         'neutral',
         '清仓后实现非正收益的持仓数量。'),
        ('清仓胜率', f"{metrics.get('win_rate', 0):.1f}%",
         'neutral',
         '盈利清仓数占全部已清仓持仓数的比例。'),
        ('平均单次清仓盈亏', _fmt_money(metrics.get('avg_profit', 0)),
         'neutral',
         '已清仓持仓的平均单笔盈亏。'),
        ('平均盈利', _fmt_money(metrics.get('avg_win', 0)),
         'neutral',
         '盈利清仓持仓的平均单笔盈利。'),
        ('平均亏损', _fmt_money(metrics.get('avg_loss', 0)),
         'neutral',
         '亏损清仓持仓的平均单笔亏损。'),
        ('最大单次盈利', _fmt_money(metrics.get('max_profit', 0)),
         'neutral',
         '已清仓持仓中的最大单笔盈利。'),
        ('最大单次亏损', _fmt_money(metrics.get('max_loss', 0)),
         'neutral',
         '已清仓持仓中的最大单笔亏损。'),
        ('总手续费', _fmt_money(metrics.get('total_commission', 0)),
         'neutral',
         '回测期间累计成交手续费。'),
        ('当前持仓平均持仓天数', f"{holding_stats.get('average_current_holding_days', 0):.2f}",
         'neutral',
         '回测结束时仍持有仓位的平均持仓交易日。'),
        ('已清仓平均持仓天数', f"{holding_stats.get('average_cleared_holding_days', 0):.2f}",
         'neutral',
         '已清仓持仓的平均持仓交易日。'),
        ('最长持仓天数', f"{holding_stats.get('max_holding_days', 0)}",
         'neutral',
         '本次回测中观测到的最长持仓交易日。'),
        ('最短持仓天数', f"{holding_stats.get('min_holding_days', 0)}",
         'neutral',
         '本次回测中观测到的最短持仓交易日。'),
    ]

    grid = ''
    for label, value, cls, tooltip in cards:
        grid += f'''
        <div class="metric-card">
            <div class="metric-label">{_make_help_label(label, tooltip)}</div>
            <div class="metric-value {cls}">{html_escape(value)}</div>
        </div>'''
    return grid



def _build_trade_stats_table(metrics: Dict) -> Dict[str, Any]:
    headers = [
        {'label': '指标', 'sort_type': 'text'},
        {'label': '数值', 'sort_type': 'text', 'align': 'right'},
    ]
    rows = [
        [_make_cell('实际买入次数', '实际买入次数'), _make_cell(str(metrics.get('executed_buy_count', 0)), metrics.get('executed_buy_count', 0))],
        [_make_cell('实际卖出次数', '实际卖出次数'), _make_cell(str(metrics.get('executed_sell_count', 0)), metrics.get('executed_sell_count', 0))],
        [_make_cell('已清仓持仓数', '已清仓持仓数'), _make_cell(str(metrics.get('cleared_positions_count', 0)), metrics.get('cleared_positions_count', 0))],
        [_make_cell('完整 round-trip 数', '完整 round-trip 数'), _make_cell(str(metrics.get('round_trip_count', 0)), metrics.get('round_trip_count', 0))],
        [_make_cell('当前持仓数', '当前持仓数'), _make_cell(str(metrics.get('current_positions_count', 0)), metrics.get('current_positions_count', 0))],
        [_make_cell('盈利清仓数', '盈利清仓数'), _make_cell(str(metrics.get('wins', 0)), metrics.get('wins', 0))],
        [_make_cell('亏损清仓数', '亏损清仓数'), _make_cell(str(metrics.get('losses', 0)), metrics.get('losses', 0))],
        [_make_cell('清仓胜率', '清仓胜率'), _make_cell(f"{metrics.get('win_rate', 0):.1f}%", metrics.get('win_rate', 0))],
        [_make_cell('平均单次清仓盈亏', '平均单次清仓盈亏'), _make_cell(_fmt_money(metrics.get('avg_profit', 0)), metrics.get('avg_profit', 0))],
        [_make_cell('平均盈利', '平均盈利'), _make_cell(_fmt_money(metrics.get('avg_win', 0)), metrics.get('avg_win', 0))],
        [_make_cell('平均亏损', '平均亏损'), _make_cell(_fmt_money(metrics.get('avg_loss', 0)), metrics.get('avg_loss', 0))],
        [_make_cell('最大单次盈利', '最大单次盈利'), _make_cell(_fmt_money(metrics.get('max_profit', 0)), metrics.get('max_profit', 0))],
        [_make_cell('最大单次亏损', '最大单次亏损'), _make_cell(_fmt_money(metrics.get('max_loss', 0)), metrics.get('max_loss', 0))],
        [_make_cell('总手续费', '总手续费'), _make_cell(_fmt_money(metrics.get('total_commission', 0)), metrics.get('total_commission', 0))],
        [_make_cell('退市归零次数', '退市归零次数'), _make_cell(str(metrics.get('delist_count', 0)), metrics.get('delist_count', 0))],
    ]
    return _make_table(headers, rows, 'trade-stats-table', '暂无交易统计', row_height=44, max_height=360)



def _build_holding_stats_table(holding_stats: Dict) -> Dict[str, Any]:
    headers = [
        {'label': '指标', 'sort_type': 'text'},
        {'label': '数值', 'sort_type': 'text', 'align': 'right'},
    ]
    rows = [
        [_make_cell('平均持仓天数', '平均持仓天数'), _make_cell(f"{holding_stats.get('average_holding_days', 0):.2f} 交易日", holding_stats.get('average_holding_days', 0))],
        [_make_cell('当前持仓平均持仓天数', '当前持仓平均持仓天数'), _make_cell(f"{holding_stats.get('average_current_holding_days', 0):.2f} 交易日", holding_stats.get('average_current_holding_days', 0))],
        [_make_cell('已清仓平均持仓天数', '已清仓平均持仓天数'), _make_cell(f"{holding_stats.get('average_cleared_holding_days', 0):.2f} 交易日", holding_stats.get('average_cleared_holding_days', 0))],
        [_make_cell('最长持仓天数', '最长持仓天数'), _make_cell(f"{holding_stats.get('max_holding_days', 0)} 交易日", holding_stats.get('max_holding_days', 0))],
        [_make_cell('最短持仓天数', '最短持仓天数'), _make_cell(f"{holding_stats.get('min_holding_days', 0)} 交易日", holding_stats.get('min_holding_days', 0))],
    ]
    return _make_table(headers, rows, 'holding-stats-table', '暂无持仓统计', row_height=44, max_height=300)


def _build_metric_details_table(metrics: Dict, holding_stats: Dict) -> Dict[str, Any]:
    headers = [
        {'label': '指标', 'sort_type': 'text'},
        {'label': '数值', 'sort_type': 'text', 'align': 'right'},
    ]
    rows = [
        [_make_cell('已清仓持仓数', '已清仓持仓数'), _make_cell(str(metrics.get('cleared_positions_count', 0)), metrics.get('cleared_positions_count', 0))],
        [_make_cell('当前持仓数', '当前持仓数'), _make_cell(str(metrics.get('current_positions_count', 0)), metrics.get('current_positions_count', 0))],
        [_make_cell('盈利清仓数', '盈利清仓数'), _make_cell(str(metrics.get('wins', 0)), metrics.get('wins', 0))],
        [_make_cell('亏损清仓数', '亏损清仓数'), _make_cell(str(metrics.get('losses', 0)), metrics.get('losses', 0))],
        [_make_cell('清仓胜率', '清仓胜率'), _make_cell(f"{metrics.get('win_rate', 0):.1f}%", metrics.get('win_rate', 0))],
        [_make_cell('平均单次清仓盈亏', '平均单次清仓盈亏'), _make_cell(_fmt_money(metrics.get('avg_profit', 0)), metrics.get('avg_profit', 0))],
        [_make_cell('平均盈利', '平均盈利'), _make_cell(_fmt_money(metrics.get('avg_win', 0)), metrics.get('avg_win', 0))],
        [_make_cell('平均亏损', '平均亏损'), _make_cell(_fmt_money(metrics.get('avg_loss', 0)), metrics.get('avg_loss', 0))],
        [_make_cell('最大单次盈利', '最大单次盈利'), _make_cell(_fmt_money(metrics.get('max_profit', 0)), metrics.get('max_profit', 0))],
        [_make_cell('最大单次亏损', '最大单次亏损'), _make_cell(_fmt_money(metrics.get('max_loss', 0)), metrics.get('max_loss', 0))],
        [_make_cell('总手续费', '总手续费'), _make_cell(_fmt_money(metrics.get('total_commission', 0)), metrics.get('total_commission', 0))],
        [_make_cell('当前持仓平均持仓天数', '当前持仓平均持仓天数'), _make_cell(f"{holding_stats.get('average_current_holding_days', 0):.2f} 交易日", holding_stats.get('average_current_holding_days', 0))],
        [_make_cell('已清仓平均持仓天数', '已清仓平均持仓天数'), _make_cell(f"{holding_stats.get('average_cleared_holding_days', 0):.2f} 交易日", holding_stats.get('average_cleared_holding_days', 0))],
        [_make_cell('最长持仓天数', '最长持仓天数'), _make_cell(f"{holding_stats.get('max_holding_days', 0)} 交易日", holding_stats.get('max_holding_days', 0))],
        [_make_cell('最短持仓天数', '最短持仓天数'), _make_cell(f"{holding_stats.get('min_holding_days', 0)} 交易日", holding_stats.get('min_holding_days', 0))],
    ]
    return _make_table(headers, rows, 'metric-details-table', '暂无指标明细', row_height=44, max_height=320)


# ---------------------------------------------------------------------------
# 主报告生成
# ---------------------------------------------------------------------------

def generate_single_report(report_data: Dict, output_dir: Path) -> Path:
    """新版单次回测报告：ECharts + TanStack Table。"""
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    html_path = output_dir / f'backtest_report_{timestamp}.html'
    stable_html_path = output_dir / 'single_report.html'

    asset_dir = Path(__file__).with_name('templates')
    styles = (asset_dir / 'single_report_styles.css').read_text(encoding='utf-8')
    module_js = (asset_dir / 'single_report_module.js').read_text(encoding='utf-8')

    config = report_data.get('individual_config', {})
    weights = config.get('weights', {})
    buy_n = config.get('buy_n', 0)
    sell_m = config.get('sell_m', 0)
    temperatures = config.get('temperatures', {})
    verify_config = report_data.get('verify_config', {}) or {}
    report_metadata = report_data.get('report_metadata', {}) or {}

    init_cash = report_data.get('init_cash', 500_000.0)
    cumulative_returns = report_data.get('cumulative_returns', []) or []
    trade_dates = report_data.get('trade_dates', []) or []
    trade_log = report_data.get('trade_log', []) or []
    daily_snapshots = report_data.get('daily_snapshots', []) or []
    positions = report_data.get('positions', []) or []
    cleared_positions = report_data.get('cleared_positions', []) or []
    delist_events = report_data.get('delist_events', []) or []
    stock_name_map = report_data.get('stock_name_map', {}) or {}
    holding_stats = report_data.get('holding_stats', {}) or {}
    per_year_metrics = report_data.get('per_year_metrics', []) or []
    period = report_data.get('period', {}) or {}
    rebalance_rule = report_data.get('rebalance_rule', {}) or {}
    final_asset = report_data.get('final_asset', init_cash)

    signal_period_str = f"{period.get('signal_start', '')} ~ {period.get('signal_end', '')}" if period else ''
    trade_period_str = f"{period.get('trade_start', '')} ~ {period.get('trade_end', '')}" if period else ''
    period_str = trade_period_str or (f"{period.get('start', '')} ~ {period.get('end', '')}" if period else '')
    trade_days = len(trade_dates)

    daily_returns_pct = _resolve_daily_returns_pct(report_data, cumulative_returns, trade_dates)
    monthly_stats = calc_monthly_stats(trade_dates, cumulative_returns, daily_returns_pct)
    strategy_nav = _cumulative_returns_to_nav(cumulative_returns)
    hs300_returns = report_data.get('hs300_returns')
    if not hs300_returns or len(hs300_returns) != len(trade_dates):
        hs300_returns = compute_hs300_cumulative_returns(trade_dates)
    hs300_nav = _cumulative_returns_to_nav(hs300_returns) if hs300_returns else []

    generated_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    metrics = compute_strategy_metrics(
        cumulative_returns_pct=cumulative_returns,
        trade_dates=trade_dates,
        init_cash=init_cash,
        final_asset=final_asset,
        trade_log=trade_log,
    )
    metrics.update({
        'executed_buy_count': report_data.get('executed_buy_count', metrics.get('buy_trades', 0)),
        'executed_sell_count': report_data.get('executed_sell_count', metrics.get('sell_trades', 0)),
        'delist_count': report_data.get('delist_count', len(delist_events)),
        'round_trip_count': report_data.get('round_trip_count', len(cleared_positions)),
        'cleared_positions_count': report_data.get('cleared_positions_count', len(cleared_positions)),
        'current_positions_count': report_data.get('current_positions_count', len(positions)),
        **holding_stats,
    })

    signal_timing = rebalance_rule.get('signal_timing', 'T-1')
    trade_timing = rebalance_rule.get('trade_timing', 'T open')
    signal_dividend_type = rebalance_rule.get('signal_dividend_type', 'back')
    execution_dividend_type = rebalance_rule.get('execution_dividend_type', signal_dividend_type)
    price_field = rebalance_rule.get('price_field', 'open')

    total_return = (final_asset - init_cash) / init_cash * 100 if init_cash else 0.0
    report_json = json.dumps({
        'summary': {
            'total_return_pct': round(total_return, 2),
            'annualized_return_pct': metrics.get('annualized', 0),
            'max_drawdown_pct': metrics.get('max_drawdown', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'calmar_ratio': metrics.get('calmar_ratio', 0),
            'win_rate_pct': metrics.get('win_rate', 0),
            'total_trades': metrics.get('total_trades', 0),
            'wins': metrics.get('wins', 0),
            'losses': metrics.get('losses', 0),
            'avg_profit': metrics.get('avg_profit', 0),
            'avg_win': metrics.get('avg_win', 0),
            'avg_loss': metrics.get('avg_loss', 0),
            'max_profit': metrics.get('max_profit', 0),
            'max_loss': metrics.get('max_loss', 0),
            'total_commission': metrics.get('total_commission', 0),
            'total_days': trade_days,
            'init_cash': init_cash,
            'final_asset': final_asset,
            'round_trips': metrics.get('round_trip_count', 0),
            'avg_holding_days': metrics.get('average_holding_days', 0),
            'delist_count': metrics.get('delist_count', 0),
        },
        'meta': {
            'period_start': period.get('start', ''),
            'period_end': period.get('end', ''),
            'signal_timing': signal_timing,
            'trade_timing': trade_timing,
            'price_field': price_field,
            'generated_time': generated_time,
        },
        'tables': {
            'monthly': _make_monthly_table(monthly_stats),
            'trades': _make_trade_table(trade_log, stock_name_map),
            'holdings': _make_holdings_table(positions, stock_name_map),
            'cleared': _make_cleared_positions_table(cleared_positions, stock_name_map),
            'delist': _make_delist_events_table(delist_events, stock_name_map),
            'daily': _make_daily_table(daily_snapshots, stock_name_map),
        },
        'charts': {
            'equity': {
                'trade_dates': trade_dates,
                'strategy_nav': strategy_nav,
                'benchmark_nav': hs300_nav,
                'daily_returns_pct': daily_returns_pct,
            },
            'distribution': _build_histogram_payload(daily_returns_pct),
            'winloss': _build_winloss_payload(trade_log),
        },
    }, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')

    metric_cards = _build_metric_cards(metrics, holding_stats)
    weights_html = ' '.join(
        f'<span class="weight-tag">{"+" if value > 0 else ""}{value:.2f} {html_escape(name)}</span>'
        for name, value in sorted(weights.items(), key=lambda item: abs(item[1]), reverse=True) if value != 0
    ) or '<span class="muted">—</span>'
    temps_html = ' '.join(
        f'<span class="temp-tag">{html_escape(name)}={value:.1f}</span>'
        for name, value in temperatures.items()
    ) or '<span class="muted">—</span>'

    verify_notice_html = ''
    if verify_config:
        verify_parts = [
            f'<strong>验证模式:</strong> {html_escape(verify_config.get("label") or "退市归零验证")}',
            f'<strong>强制优先股票:</strong> {html_escape(verify_config.get("force_stock_code", ""))}',
        ]
        candidate_codes = verify_config.get('candidate_stock_codes') or []
        if candidate_codes:
            verify_parts.append(f'<strong>候选股票池:</strong> {html_escape(", ".join(candidate_codes))}')
        stock_pool_size = report_metadata.get('stock_pool_size')
        if stock_pool_size is not None:
            verify_parts.append(f'<strong>实际 TopN 股票池:</strong> {stock_pool_size} 只')
        config_path = report_metadata.get('config_path')
        if config_path:
            verify_parts.append(f'<strong>配置文件:</strong> {html_escape(config_path)}')
        if verify_config.get('notes'):
            verify_parts.append(f'<strong>备注:</strong> {html_escape(verify_config["notes"])}')
        verify_notice_html = '<div class="verify-box">' + '<br>'.join(verify_parts) + '</div>'

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>回测报告 - {html_escape(period_str)}</title><link rel="icon" href="data:,"><link rel="stylesheet" href="https://unpkg.com/tippy.js@6/dist/tippy.css"><script src="https://unpkg.com/@popperjs/core@2/dist/umd/popper.min.js"></script><script src="https://unpkg.com/tippy.js@6/dist/tippy-bundle.umd.min.js"></script><script src="https://cdn.jsdelivr.net/npm/echarts@6.0.0/dist/echarts.min.js"></script><style>{styles}</style></head>
<body><div class="container">
<div class="report-header"><div class="report-title">回测报告</div><div class="report-subtitle">WBR 量化交易系统 · T-1 信号 / T 日开盘执行 详细报告</div><div class="report-meta"><span>信号周期: {html_escape(signal_period_str)}</span><span>执行周期: {html_escape(trade_period_str or period_str)}</span><span>调仓日数: {trade_days}</span><span>初始资金: {_fmt_money(init_cash)}</span><span>最终资产: {_fmt_money(final_asset)}</span><span>生成时间: {html_escape(generated_time)}</span></div></div>
<div class="config-box"><strong>因子权重:</strong> {weights_html}<br><strong>温度参数:</strong> {temps_html}<br><strong>调仓配置:</strong> buy_n={buy_n}, sell_m={sell_m}<br><strong>调仓规则:</strong> 信号={html_escape(signal_timing)} &nbsp;|&nbsp; 执行={html_escape(trade_timing)} &nbsp;|&nbsp; 价格字段={html_escape(price_field)}<br><strong>复权口径:</strong> 信号={html_escape(signal_dividend_type)} &nbsp;→&nbsp; 执行={html_escape(execution_dividend_type)} &nbsp;|&nbsp;<strong>实际买入次数:</strong> {metrics.get('executed_buy_count', 0)} &nbsp;|&nbsp;<strong>实际卖出次数:</strong> {metrics.get('executed_sell_count', 0)} &nbsp;|&nbsp;<strong>完整 round-trip 数:</strong> {metrics.get('round_trip_count', 0)}</div>
{verify_notice_html}
<div class="card"><div class="card-title">核心指标</div><div class="metrics-grid">{metric_cards}</div></div>
<div class="card"><div class="card-title">分年度指标</div>{_build_per_year_table(per_year_metrics)}</div>
<div class="card"><div class="card-title">净值曲线</div><div id="equity-chart" class="chart-lg"></div></div>
<div class="charts-2col"><div class="card"><div class="card-title">每日收益率分布</div><div id="distribution-chart" class="chart"></div></div><div class="card"><div class="card-title">盈亏分布</div><div id="winloss-chart" class="chart"></div></div></div>
<div class="card"><div class="card-title">月度收益</div><div id="monthly-host"></div></div>
<h2>交易记录明细</h2><div class="card"><div id="trade-host"></div></div>
<h2>当前持仓明细 ({len(positions)} 只)</h2><div class="card"><div id="holdings-host"></div></div>
<h2>已清仓持仓明细 ({len(cleared_positions)} 只)</h2><div class="card"><div id="cleared-host"></div></div>
<h2>每日资金快照</h2><div class="card"><div id="daily-host"></div></div>
<h2>退市归零事件 ({len(delist_events)} 只)</h2><div class="card"><div id="delist-host"></div></div>
<div class="footer">WBR 量化交易系统 · 回测报告 · 实际成交 {len(trade_log)} 笔 · {trade_days} 个交易日</div></div>
<script id="report-data" type="application/json">{report_json}</script><script type="module">{module_js}</script></body></html>"""

    minified_html = _minify_html_document(html_content)

    with open(html_path, 'w', encoding='utf-8') as file_obj:
        file_obj.write(minified_html)
    with open(stable_html_path, 'w', encoding='utf-8') as file_obj:
        file_obj.write(minified_html)

    testback_logger.info(f'详细回测报告已生成: {html_path}')
    testback_logger.info(f'单次回测报告已更新: {stable_html_path}')
    return stable_html_path
