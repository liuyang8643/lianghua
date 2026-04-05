"""
WBR 回测可视化报告生成器 - 增强版

包含：
1. 策略概览（配置参数）
2. 核心指标（收益率、年化、夏普、最大回撤、卡玛等）
3. 累计收益曲线（vs 沪深300基准）
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



def _make_stock_text(code: str, stock_name_map: Dict[str, str]) -> str:
    return f'<span class="stock-text">{html_escape(_fmt_stock(code, stock_name_map))}</span>'



def _make_stock_button(code: str, stock_name_map: Dict[str, str], class_name: str = 'code-btn') -> str:
    label = html_escape(_fmt_stock(code, stock_name_map))
    return f'<button class="{class_name}" onclick="showKline(\'{html_escape(code)}\')">{label}</button>'



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



def _make_cell(html: str, sort_value: Any = None, cell_class: str = '') -> Dict[str, Any]:
    return {
        'html': html,
        'sort': '' if sort_value is None else str(sort_value),
        'class': cell_class,
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
        from core.database import get_market_data_from_cache
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

    if data is None or data.empty or len(data) < 5:
        return ''

    timestamps = data['time'].values
    dates = [datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d') for ts in timestamps]
    opens = data['open'].values
    highs = data['high'].values
    lows = data['low'].values
    closes = data['close'].values
    volumes = data['amount'].values

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



def _build_all_kline_section(trade_log: List[Dict], trade_dates: List[str],
                             stock_name_map: Dict[str, str]) -> str:
    """为所有交易过的股票生成K线图表区段"""
    if not trade_log:
        return ''

    traded_codes = sorted(set(t['code'] for t in trade_log if t.get('code')))
    if not traded_codes:
        return ''

    chart_sections = []
    button_sections = []

    for code in traded_codes:
        chart_html = _build_kline_chart(code, trade_log, trade_dates, stock_name_map)
        display_text = html_escape(_fmt_stock(code, stock_name_map))
        has_kline = bool(chart_html)
        if has_kline:
            chart_sections.append(f'''
        <div id="kline-{code.replace('.', '-')}" class="kline-chart" style="display:none;">
            <div class="kline-header">
                <span class="kline-code">{display_text}</span>
                <button class="kline-close" onclick="closeKline()">× 关闭</button>
            </div>
            {chart_html}
        </div>''')

        cls = 'stock-pill' if has_kline else 'stock-pill no-data'
        onclick = f"showKline('{code}')" if has_kline else ''
        button_sections.append(f'<button class="{cls}" onclick="{onclick}">{display_text}</button>')

    if not chart_sections:
        return ''

    buttons_html = '<div class="kline-stock-pills">' + ''.join(button_sections) + '</div>'
    kline_divs = ''.join(chart_sections)

    return f'''
    <div class="kline-overlay" id="klineOverlay" onclick="closeKline()"></div>
    <div class="kline-panel" id="klinePanel">
        <div class="kline-panel-header">
            <span>K线走势（点击股票查看）</span>
            <button class="panel-close" onclick="closeKline()">×</button>
        </div>
        {buttons_html}
        <div class="kline-charts-area" id="klineChartsArea">
            <div class="kline-placeholder" id="klinePlaceholder">点击上方股票按钮查看 K 线图</div>
            {kline_divs}
        </div>
    </div>'''



def _collect_kline_raw_data(trade_log: List[Dict], trade_dates: List[str],
                            stock_name_map: Dict[str, str]) -> Dict:
    """收集所有交易过股票的原始 OHLCV 数据（用于前端按需渲染）。"""
    if not trade_log or not trade_dates:
        return {}

    try:
        from core.database import get_market_data_from_cache
    except ImportError:
        return {}

    traded_codes = sorted(set(t['code'] for t in trade_log if t.get('code')))
    kline_data: Dict[str, Any] = {}

    for code in traded_codes:
        try:
            last_date = datetime.strptime(trade_dates[-1], '%Y-%m-%d')
            end_dt = last_date + timedelta(days=60)
            data = get_market_data_from_cache(code, 300, end_dt, '1d', dividend_type='back')
            if data is None or data.empty or len(data) < 5:
                continue

            timestamps = data['time'].values
            dates = [datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d') for ts in timestamps]
            stock_trades = _get_stock_trades(trade_log, code)

            kline_data[code] = {
                'n': _get_stock_name(code, stock_name_map),
                'd': dates,
                'o': [round(float(v), 2) for v in data['open'].values],
                'h': [round(float(v), 2) for v in data['high'].values],
                'l': [round(float(v), 2) for v in data['low'].values],
                'c': [round(float(v), 2) for v in data['close'].values],
                'v': [int(float(v)) for v in data['amount'].values],
                'b': [{'d': b['trade_date'], 'p': round(b['price'], 4)} for b in stock_trades['buys']],
                's': [{'d': s['trade_date'], 'p': round(s['price'], 4)} for s in stock_trades['sells']],
            }
        except Exception:
            continue

    return kline_data



def _build_lazy_kline_section(trade_log: List[Dict], trade_dates: List[str],
                              stock_name_map: Dict[str, str]) -> str:
    """构建 K 线区段：数据以 gzip+base64 压缩后嵌入 HTML，点击时才用 Plotly 渲染。

    相比 _build_all_kline_section 预渲染全部图表（~37 MB），本方法压缩后仅 ~5 MB，
    且浏览器无需在加载时创建上千个 Plotly 图表实例。
    """
    import gzip as _gzip
    import base64 as _base64
    import json as _json

    kline_data = _collect_kline_raw_data(trade_log, trade_dates, stock_name_map)
    if not kline_data:
        return ''

    json_str = _json.dumps(kline_data, ensure_ascii=False, separators=(',', ':'))
    compressed = _gzip.compress(json_str.encode('utf-8'), compresslevel=9)
    b64_data = _base64.b64encode(compressed).decode('ascii')

    testback_logger.info(
        f'K线数据: {len(kline_data)} 只股票, '
        f'原始 {len(json_str)/1024/1024:.1f} MB → '
        f'压缩后 {len(b64_data)/1024/1024:.1f} MB'
    )

    traded_codes = sorted(kline_data.keys())
    buttons = []
    for code in traded_codes:
        name = kline_data[code].get('n', '')
        label = f'{code} {name}' if name else code
        buttons.append(
            f'<button class="stock-pill" onclick="showKline(\'{code}\')">'
            f'{html_escape(label)}</button>'
        )
    buttons_html = '<div class="kline-stock-pills">' + ''.join(buttons) + '</div>'

    return f'''
    <div class="kline-overlay" id="klineOverlay" onclick="closeKline()"></div>
    <div class="kline-panel" id="klinePanel">
        <div class="kline-panel-header">
            <span>K线走势（点击股票查看，共 {len(traded_codes)} 只）</span>
            <button class="panel-close" onclick="closeKline()">×</button>
        </div>
        {buttons_html}
        <div class="kline-charts-area" id="klineChartsArea">
            <div class="kline-placeholder" id="klinePlaceholder">
                点击上方股票按钮查看 K 线图
            </div>
            <div id="klineRenderArea" style="display:none;padding:8px;"></div>
        </div>
    </div>
    <script>var KLINE_B64="{b64_data}";</script>'''


# ---------------------------------------------------------------------------
# 基准数据
# ---------------------------------------------------------------------------


def get_hs300_daily_returns(trade_dates: List[str]) -> List[float]:
    """兼容旧调用：委托到统一指标模块。"""
    return compute_hs300_cumulative_returns(trade_dates)


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------


def calc_metrics(daily_returns_pct: List[float], trade_dates: List[str],
                 init_cash: float, final_asset: float,
                 trade_log: List[Dict]) -> Dict:
    """兼容旧调用：委托到统一指标模块。"""
    return compute_strategy_metrics(
        cumulative_returns_pct=daily_returns_pct,
        trade_dates=trade_dates,
        init_cash=init_cash,
        final_asset=final_asset,
        trade_log=trade_log,
    )



def calc_monthly_stats(trade_dates: List[str], daily_returns: List[float]) -> List[Dict]:
    """计算月度收益统计"""
    if not trade_dates or not daily_returns:
        return []

    monthly: Dict[str, dict] = {}
    for i, d in enumerate(trade_dates):
        ym = d[:7]
        if ym not in monthly:
            monthly[ym] = {'count': 0, 'returns': [], 'win_days': 0, 'lose_days': 0}
        monthly[ym]['count'] += 1
        monthly[ym]['returns'].append(daily_returns[i])
        if daily_returns[i] > 0:
            monthly[ym]['win_days'] += 1
        elif daily_returns[i] < 0:
            monthly[ym]['lose_days'] += 1

    result = []
    for ym in sorted(monthly.keys()):
        m = monthly[ym]
        rets = m['returns']
        if rets:
            monthly_ret = round(rets[-1] - rets[0], 2)
            cum_ret = round(rets[-1], 2)
            result.append({
                'month': ym,
                'monthly_return': monthly_ret,
                'cumulative_return': cum_ret,
                'trade_days': m['count'],
                'win_days': m['win_days'],
                'lose_days': m['lose_days'],
                'win_rate': round(m['win_days'] / m['count'] * 100, 1) if m['count'] > 0 else 0,
            })
    return result


# ---------------------------------------------------------------------------
# 表格生成
# ---------------------------------------------------------------------------


def _make_table(headers: List[Any], rows: List[List[Any]], classes: str = 'data-table sortable-table',
                id_attr: str = '') -> str:
    """生成 HTML 表格，支持 data-sort 排序元数据。"""
    id_html = f' id="{id_attr}"' if id_attr else ''

    header_html = ''
    for header in headers:
        if isinstance(header, dict):
            label = header.get('label', '')
            sort_type = header.get('sort_type', 'text')
            sortable = header.get('sortable', True)
        else:
            label = str(header)
            sort_type = 'text'
            sortable = True
        th_class = 'sortable' if sortable else ''
        sort_attr = 'true' if sortable else 'false'
        header_html += (
            f'<th class="{th_class}" data-sortable="{sort_attr}" '
            f'data-sort-type="{sort_type}">{html_escape(label)}</th>'
        )
    thead = f'<thead><tr>{header_html}</tr></thead>'

    tbody_rows = ''
    for row in rows:
        cells = ''
        for cell in row:
            if isinstance(cell, dict):
                cell_html = cell.get('html', '')
                sort_value = html_escape(str(cell.get('sort', '')))
                cell_class = cell.get('class', '')
            else:
                cell_html = str(cell)
                sort_value = html_escape(str(cell))
                cell_class = ''
            class_attr = f' class="{cell_class}"' if cell_class else ''
            cells += f'<td{class_attr} data-sort="{sort_value}">{cell_html}</td>'
        tbody_rows += f'<tr>{cells}</tr>'
    tbody = f'<tbody>{tbody_rows}</tbody>'
    return f'<table{id_html} class="{classes}">{thead}{tbody}</table>'



def _make_daily_table(daily_snapshots: List[Dict], stock_name_map: Dict[str, str]) -> str:
    if not daily_snapshots:
        return '<p class="no-data">暂无每日快照数据</p>'

    headers = [
        {'label': '信号日', 'sort_type': 'date'},
        {'label': '执行日', 'sort_type': 'date'},
        {'label': '执行基准', 'sort_type': 'text'},
        {'label': '现金(¥)', 'sort_type': 'number'},
        {'label': '持仓市值(¥)', 'sort_type': 'number'},
        {'label': '总资产(¥)', 'sort_type': 'number'},
        {'label': '日收益率', 'sort_type': 'number'},
        {'label': '累计收益率', 'sort_type': 'number'},
        {'label': '候选 buy_n 列表', 'sort_type': 'text'},
        {'label': '候选 buy_n-diff', 'sort_type': 'text'},
        {'label': '实际买卖', 'sort_type': 'text'},
    ]
    rows = []
    for s in daily_snapshots:
        daily_pct = s.get('daily_return_pct', 0)
        cum_pct = s.get('cumulative_return_pct', 0)
        daily_cls = 'pos-cell' if daily_pct > 0 else ('neg-cell' if daily_pct < 0 else '')
        cum_cls = 'pos-cell' if cum_pct > 0 else ('neg-cell' if cum_pct < 0 else '')
        buy_n_list = s.get('buy_n_list', [])
        buy_n_diff_list = s.get('buy_n_diff_list', [code for code in buy_n_list if code not in s.get('sell_m_list', [])])
        executed_buy_list = s.get('executed_buy_list', [])
        executed_sell_list = s.get('executed_sell_list', [])
        signal_date = _get_signal_date(s)
        trade_date = _get_trade_date(s)
        execution_basis = _format_execution_basis(s)
        action_sort = '|'.join(
            [f'B:{_fmt_stock(code, stock_name_map)}' for code in executed_buy_list] +
            [f'S:{_fmt_stock(code, stock_name_map)}' for code in executed_sell_list]
        )
        rows.append([
            _make_cell(html_escape(signal_date), signal_date),
            _make_cell(html_escape(trade_date), trade_date),
            _make_cell(html_escape(execution_basis), execution_basis),
            _make_cell(f'{s["cash"]:,.2f}', s['cash']),
            _make_cell(f'{s["market_value"]:,.2f}', s['market_value']),
            _make_cell(f'{s["total_asset"]:,.2f}', s['total_asset']),
            _make_cell(f'<span class="{daily_cls}">{_fmt_pct(daily_pct)}</span>', daily_pct),
            _make_cell(f'<span class="{cum_cls}">{_fmt_pct(cum_pct)}</span>', cum_pct),
            _make_cell(_make_stock_list_html(buy_n_list, stock_name_map), '|'.join(_fmt_stock(code, stock_name_map) for code in buy_n_list)),
            _make_cell(_make_stock_list_html(buy_n_diff_list, stock_name_map), '|'.join(_fmt_stock(code, stock_name_map) for code in buy_n_diff_list)),
            _make_cell(_make_stock_action_list_html(executed_buy_list, executed_sell_list, stock_name_map), action_sort),
        ])
    return _make_table(headers, rows, 'data-table sortable-table', 'daily-table')



def _make_trade_table(trade_log: List[Dict], stock_name_map: Dict[str, str]) -> str:
    if not trade_log:
        return '<p class="no-data">暂无交易记录</p>'

    headers = [
        {'label': '信号日', 'sort_type': 'date'},
        {'label': '执行日', 'sort_type': 'date'},
        {'label': '股票', 'sort_type': 'text'},
        {'label': '方向', 'sort_type': 'text'},
        {'label': '执行基准', 'sort_type': 'text'},
        {'label': '价格(¥)', 'sort_type': 'number'},
        {'label': '数量(股)', 'sort_type': 'number'},
        {'label': '金额(¥)', 'sort_type': 'number'},
        {'label': '手续费(¥)', 'sort_type': 'number'},
        {'label': '盈亏(¥)', 'sort_type': 'number'},
        {'label': '说明', 'sort_type': 'text'},
    ]
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
    return _make_table(headers, rows, 'data-table sortable-table trade-table', 'trade-table')



def _make_holdings_table(positions: List[Dict], stock_name_map: Dict[str, str]) -> str:
    if not positions:
        return '<p class="no-data">当前无持仓</p>'

    headers = [
        {'label': '股票', 'sort_type': 'text'},
        {'label': '信号日', 'sort_type': 'date'},
        {'label': '执行日', 'sort_type': 'date'},
        {'label': '执行基准', 'sort_type': 'text'},
        {'label': '持仓天数(交易日)', 'sort_type': 'number'},
        {'label': '持仓数量', 'sort_type': 'number'},
        {'label': '持仓均价(¥)', 'sort_type': 'number'},
        {'label': '持仓成本(¥)', 'sort_type': 'number'},
        {'label': '现价(¥)', 'sort_type': 'number'},
        {'label': '当前市值(¥)', 'sort_type': 'number'},
        {'label': '盈亏(¥)', 'sort_type': 'number'},
        {'label': '盈亏率', 'sort_type': 'number'},
    ]
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
    return _make_table(headers, rows, 'data-table sortable-table', 'holdings-table')



def _make_cleared_positions_table(cleared_positions: List[Dict], stock_name_map: Dict[str, str]) -> str:
    if not cleared_positions:
        return '<p class="no-data">暂无已清仓持仓</p>'

    headers = [
        {'label': '股票', 'sort_type': 'text'},
        {'label': '买入信号日', 'sort_type': 'date'},
        {'label': '买入执行日', 'sort_type': 'date'},
        {'label': '卖出信号日', 'sort_type': 'date'},
        {'label': '卖出执行日', 'sort_type': 'date'},
        {'label': '执行基准', 'sort_type': 'text'},
        {'label': '持仓天数(交易日)', 'sort_type': 'number'},
        {'label': '数量(股)', 'sort_type': 'number'},
        {'label': '买入均价(¥)', 'sort_type': 'number'},
        {'label': '卖出价格(¥)', 'sort_type': 'number'},
        {'label': '持仓成本(¥)', 'sort_type': 'number'},
        {'label': '盈亏(¥)', 'sort_type': 'number'},
        {'label': '盈亏率', 'sort_type': 'number'},
        {'label': '清仓原因', 'sort_type': 'text'},
    ]
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
    return _make_table(headers, rows, 'data-table sortable-table', 'cleared-table')



def _make_delist_events_table(delist_events: List[Dict], stock_name_map: Dict[str, str]) -> str:
    if not delist_events:
        return '<p class="no-data">暂无退市归零事件</p>'

    headers = [
        {'label': '股票', 'sort_type': 'text'},
        {'label': '退市日', 'sort_type': 'date'},
        {'label': '归零信号日', 'sort_type': 'date'},
        {'label': '归零执行日', 'sort_type': 'date'},
        {'label': '买入执行日', 'sort_type': 'date'},
        {'label': '持仓天数(交易日)', 'sort_type': 'number'},
        {'label': '数量(股)', 'sort_type': 'number'},
        {'label': '持仓成本(¥)', 'sort_type': 'number'},
        {'label': '归零损失(¥)', 'sort_type': 'number'},
        {'label': '损失率', 'sort_type': 'number'},
        {'label': '说明', 'sort_type': 'text'},
    ]
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
    return _make_table(headers, rows, 'data-table sortable-table', 'delist-table')



def _make_monthly_table(monthly_stats: List[Dict]) -> str:
    if not monthly_stats:
        return '<p class="no-data">暂无月度数据</p>'

    headers = [
        {'label': '月份', 'sort_type': 'text'},
        {'label': '月收益率', 'sort_type': 'number'},
        {'label': '累计收益率', 'sort_type': 'number'},
        {'label': '交易天数', 'sort_type': 'number'},
        {'label': '盈利天数', 'sort_type': 'number'},
        {'label': '亏损天数', 'sort_type': 'number'},
        {'label': '日胜率', 'sort_type': 'number'},
    ]
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
    return _make_table(headers, rows, 'data-table sortable-table', 'monthly-table')


# ---------------------------------------------------------------------------
# Plotly 图表生成
# ---------------------------------------------------------------------------


def _build_equity_chart(trade_dates: List[str], cumulative_returns: List[float],
                        hs300_returns: List[float]) -> str:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return '<p class="no-data">需要安装 plotly: pip install plotly</p>'

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=('累计收益率曲线', '每日收益率')
    )

    fig.add_trace(go.Scatter(
        x=trade_dates,
        y=cumulative_returns,
        mode='lines',
        name='策略',
        line=dict(color='#1976D2', width=2),
        hovertemplate='日期: %{x}<br>累计收益: %{y:.2f}%<extra></extra>',
    ), row=1, col=1)

    if hs300_returns and len(hs300_returns) == len(trade_dates):
        fig.add_trace(go.Scatter(
            x=trade_dates,
            y=hs300_returns,
            mode='lines',
            name='沪深300',
            line=dict(color='#FF9800', width=1.5, dash='dash'),
            hovertemplate='日期: %{x}<br>累计收益: %{y:.2f}%<extra></extra>',
        ), row=1, col=1)

    if len(cumulative_returns) > 1:
        daily_pct = np.diff(cumulative_returns).tolist()
        colors = ['#C62828' if v < 0 else '#2E7D32' for v in daily_pct]
        fig.add_trace(go.Bar(
            x=trade_dates[1:],
            y=daily_pct,
            name='日收益率',
            marker_color=colors,
            hovertemplate='日期: %{x}<br>日收益: %{y:.2f}%<extra></extra>',
        ), row=2, col=1)
        fig.add_hline(y=0, line_dash='dot', line_color='gray', line_width=1, row=2, col=1)

    fig.update_layout(
        height=480,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified',
        showlegend=True,
    )
    fig.update_xaxes(title_text='日期', row=2, col=1)
    fig.update_yaxes(title_text='累计收益率 (%)', row=1, col=1)
    fig.update_yaxes(title_text='日收益率 (%)', row=2, col=1)

    return fig.to_html(full_html=False, include_plotlyjs=False)



def _build_distribution_chart(daily_returns: List[float]) -> str:
    try:
        import plotly.graph_objects as go
    except ImportError:
        return ''

    if not daily_returns or len(daily_returns) < 2:
        return ''

    cumulative = np.array(daily_returns)
    daily_pct = np.diff(cumulative).tolist()
    mean_val = round(float(np.mean(daily_pct)), 4)
    std_val = round(float(np.std(daily_pct, ddof=1)), 4)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=daily_pct,
        nbinsx=40,
        name='日收益率分布',
        marker_color='#42A5F5',
        hovertemplate='收益率: %{x:.3f}%<br>次数: %{y}<extra></extra>',
    ))
    fig.add_vline(x=mean_val, line_color='#4CAF50', line_width=2, line_dash='dash',
                  annotation_text=f'均值: {mean_val:.3f}%', annotation_position='top right')
    fig.add_vline(x=0, line_color='gray', line_width=1, line_dash='dot')

    fig.update_layout(
        title=dict(text=f'每日收益率分布 (均值={mean_val:.3f}%, σ={std_val:.3f}%)', x=0.5, font=dict(size=14)),
        xaxis_title='每日收益率 (%)',
        yaxis_title='频次',
        height=300,
        bargap=0.1,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)



def _build_winloss_chart(trade_log: List[Dict]) -> str:
    try:
        import plotly.graph_objects as go
    except ImportError:
        return ''

    sell_trades = [t for t in trade_log if t.get('action') == 'sell' and t.get('income') is not None]
    if not sell_trades:
        return ''

    wins = len([t for t in sell_trades if t.get('income', 0) > 0])
    losses = len([t for t in sell_trades if t.get('income', 0) <= 0])
    fig = go.Figure(data=[go.Pie(
        labels=['盈利清仓', '亏损清仓'],
        values=[wins, losses],
        hole=0.45,
        marker_colors=['#2E7D32', '#C62828'],
    )])
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
    return fig.to_html(full_html=False, include_plotlyjs=False)


# ---------------------------------------------------------------------------
# 核心指标卡片 / 统计表
# ---------------------------------------------------------------------------


def _build_metric_cards(metrics: Dict) -> str:
    cards = [
        ('总收益率', _fmt_pct(metrics.get('total_return', 0)), 'pos' if metrics.get('total_return', 0) >= 0 else 'neg'),
        ('年化收益率', _fmt_pct(metrics.get('annualized', 0)), 'pos' if metrics.get('annualized', 0) >= 0 else 'neg'),
        ('最大回撤', _fmt_pct(metrics.get('max_drawdown', 0), sign=False), 'neg'),
        ('夏普比率', f"{metrics.get('sharpe_ratio', 0):.2f}", 'neutral'),
        ('实际买入次数', str(metrics.get('executed_buy_count', 0)), 'neutral'),
        ('实际卖出次数', str(metrics.get('executed_sell_count', 0)), 'neutral'),
        ('退市归零次数', str(metrics.get('delist_count', 0)), 'neutral'),
        ('完整 round-trip', str(metrics.get('round_trip_count', 0)), 'neutral'),
        ('平均持仓天数', f"{metrics.get('average_holding_days', 0):.2f}", 'neutral'),
    ]

    grid = ''
    for label, value, cls in cards:
        grid += f'''
        <div class="metric-card">
            <div class="metric-label">{html_escape(label)}</div>
            <div class="metric-value {cls}">{html_escape(value)}</div>
        </div>'''
    return grid



def _build_trade_stats_table(metrics: Dict) -> str:
    rows = [
        ('实际买入次数', str(metrics.get('executed_buy_count', 0))),
        ('实际卖出次数', str(metrics.get('executed_sell_count', 0))),
        ('退市归零次数', str(metrics.get('delist_count', 0))),
        ('已清仓持仓数', str(metrics.get('cleared_positions_count', 0))),
        ('完整 round-trip 数', str(metrics.get('round_trip_count', 0))),
        ('当前持仓数', str(metrics.get('current_positions_count', 0))),
        ('盈利清仓数', str(metrics.get('wins', 0))),
        ('亏损清仓数', str(metrics.get('losses', 0))),
        ('清仓胜率', f"{metrics.get('win_rate', 0):.1f}%"),
        ('平均单次清仓盈亏', _fmt_money(metrics.get('avg_profit', 0))),
        ('平均盈利', _fmt_money(metrics.get('avg_win', 0))),
        ('平均亏损', _fmt_money(metrics.get('avg_loss', 0))),
        ('最大单次盈利', _fmt_money(metrics.get('max_profit', 0))),
        ('最大单次亏损', _fmt_money(metrics.get('max_loss', 0))),
        ('总手续费', _fmt_money(metrics.get('total_commission', 0))),
    ]
    html = '<table class="stats-table">'
    for k, v in rows:
        html += f'<tr><th>{html_escape(k)}</th><td>{html_escape(v)}</td></tr>'
    html += '</table>'
    return html



def _build_holding_stats_table(holding_stats: Dict) -> str:
    rows = [
        ('平均持仓天数', f"{holding_stats.get('average_holding_days', 0):.2f} 交易日"),
        ('当前持仓平均持仓天数', f"{holding_stats.get('average_current_holding_days', 0):.2f} 交易日"),
        ('已清仓平均持仓天数', f"{holding_stats.get('average_cleared_holding_days', 0):.2f} 交易日"),
        ('最长持仓天数', f"{holding_stats.get('max_holding_days', 0)} 交易日"),
        ('最短持仓天数', f"{holding_stats.get('min_holding_days', 0)} 交易日"),
    ]
    html = '<table class="stats-table">'
    for k, v in rows:
        html += f'<tr><th>{html_escape(k)}</th><td>{html_escape(v)}</td></tr>'
    html += '</table>'
    return html


# ---------------------------------------------------------------------------
# 主报告生成
# ---------------------------------------------------------------------------


def generate_single_report(report_data: Dict, output_dir: Path) -> Path:
    """生成详细的回测报告"""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    html_path = output_dir / f'backtest_report_{timestamp}.html'
    stable_html_path = output_dir / 'single_report.html'

    try:
        import plotly.graph_objects as go  # noqa: F401
    except ImportError:
        testback_logger.error('需要安装 plotly: pip install plotly')
        raise

    config = report_data.get('individual_config', {})
    weights = config.get('weights', {})
    buy_n = config.get('buy_n', 0)
    sell_m = config.get('sell_m', 0)
    temperatures = config.get('temperatures', {})
    freeze_days = config.get('freeze_days', 0)
    verify_config = report_data.get('verify_config', {}) or {}
    report_metadata = report_data.get('report_metadata', {}) or {}

    init_cash = report_data.get('init_cash', 500_000.0)
    cumulative_returns = report_data.get('cumulative_returns', [])
    daily_returns_raw = report_data.get('daily_returns', [])
    signal_dates = report_data.get('signal_dates', [])
    trade_dates = report_data.get('trade_dates', [])
    trade_log = report_data.get('trade_log', [])
    daily_snapshots = report_data.get('daily_snapshots', [])
    positions = report_data.get('positions', [])
    cleared_positions = report_data.get('cleared_positions', [])
    delist_events = report_data.get('delist_events', [])
    stock_name_map = report_data.get('stock_name_map', {})
    holding_stats = report_data.get('holding_stats', {})
    period = report_data.get('period', {})
    rebalance_rule = report_data.get('rebalance_rule', {})
    final_asset = report_data.get('final_asset', init_cash)

    signal_period_str = f"{period.get('signal_start', '')} ~ {period.get('signal_end', '')}" if period else ''
    trade_period_str = f"{period.get('trade_start', '')} ~ {period.get('trade_end', '')}" if period else ''
    period_str = trade_period_str or (f"{period.get('start', '')} ~ {period.get('end', '')}" if period else '')
    trade_days = len(trade_dates)

    metrics = report_data.get('metrics') or calc_metrics(
        daily_returns_raw, trade_dates, init_cash, final_asset, trade_log
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

    monthly_stats = calc_monthly_stats(trade_dates, daily_returns_raw)
    hs300_returns = report_data.get('hs300_returns')
    if not hs300_returns:
        hs300_returns = get_hs300_daily_returns(trade_dates)

    equity_chart = _build_equity_chart(trade_dates, cumulative_returns, hs300_returns)
    dist_chart = _build_distribution_chart(cumulative_returns)
    winloss_chart = _build_winloss_chart(trade_log)
    trade_table = _make_trade_table(trade_log, stock_name_map)
    holdings_table = _make_holdings_table(positions, stock_name_map)
    cleared_table = _make_cleared_positions_table(cleared_positions, stock_name_map)
    delist_table = _make_delist_events_table(delist_events, stock_name_map)
    daily_table = _make_daily_table(daily_snapshots, stock_name_map)
    monthly_table = _make_monthly_table(monthly_stats)
    metric_cards = _build_metric_cards(metrics)
    trade_stats_html = _build_trade_stats_table(metrics)
    holding_stats_html = _build_holding_stats_table(holding_stats)
    kline_section = _build_lazy_kline_section(trade_log, trade_dates, stock_name_map)

    weights_html = ' '.join(
        f'<span class="weight-tag">{"+" if v > 0 else ""}{v:.2f} {html_escape(k)}</span>'
        for k, v in sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True) if v != 0
    ) or '<span class="muted">—</span>'
    temps_html = ' '.join(
        f'<span class="temp-tag">{html_escape(k)}={v:.1f}</span>'
        for k, v in temperatures.items()
    ) or '<span class="muted">—</span>'
    signal_timing = rebalance_rule.get('signal_timing', 'T-1')
    trade_timing = rebalance_rule.get('trade_timing', 'T open')
    signal_dividend_type = rebalance_rule.get('signal_dividend_type', 'back')
    execution_dividend_type = rebalance_rule.get('execution_dividend_type', signal_dividend_type)
    price_field = rebalance_rule.get('price_field', 'open')

    verify_notice_html = ''
    if verify_config:
        verify_parts = [
            f'<strong>验证模式:</strong> {html_escape(verify_config.get("label") or "退市归零验证")}',
            f'<strong>强制优先股票:</strong> {html_escape(verify_config.get("force_stock_code", ""))}',
        ]
        candidate_codes = verify_config.get('candidate_stock_codes') or []
        if candidate_codes:
            verify_parts.append(
                f'<strong>候选股票池:</strong> {html_escape(", ".join(candidate_codes))}'
            )
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
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>回测报告 - {html_escape(period_str)}</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
    <style>
        :root {{
            --bg: #f5f6fa;
            --card-bg: #ffffff;
            --text: #212529;
            --text-muted: #6c757d;
            --border: #dee2e6;
            --positive: #2E7D32;
            --negative: #C62828;
            --accent: #1976D2;
            --accent-light: #e3f2fd;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 24px;
            line-height: 1.6;
            font-size: 14px;
        }}
        .container {{ max-width: 1480px; margin: 0 auto; }}
        .report-header {{ margin-bottom: 24px; }}
        .report-title {{ font-size: 26px; font-weight: 700; color: var(--text); }}
        .report-subtitle {{ color: var(--text-muted); font-size: 14px; margin-top: 4px; }}
        .report-meta {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 12px; font-size: 13px; color: var(--text-muted); }}
        .report-meta span {{ background: var(--card-bg); padding: 4px 12px; border-radius: 16px; border: 1px solid var(--border); }}

        .card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 16px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.07);
        }}
        .card-title {{
            font-size: 15px;
            font-weight: 600;
            color: var(--text);
            border-bottom: 2px solid var(--accent);
            padding-bottom: 8px;
            margin-bottom: 16px;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px;
        }}
        .metric-card {{
            background: var(--card-bg);
            border-radius: 10px;
            padding: 16px 12px;
            text-align: center;
            box-shadow: 0 1px 4px rgba(0,0,0,0.07);
        }}
        .metric-label {{ font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }}
        .metric-value {{ font-size: 22px; font-weight: 700; line-height: 1.2; }}
        .metric-value.pos {{ color: var(--positive); }}
        .metric-value.neg {{ color: var(--negative); }}
        .metric-value.neutral {{ color: var(--text); }}

        .config-box {{ background: var(--accent-light); border-radius: 10px; padding: 16px 20px; margin-bottom: 16px; font-size: 13px; }}
        .config-box strong {{ color: var(--accent); }}
        .verify-box {{ background: #fff8e1; border: 1px solid #ffd54f; border-radius: 10px; padding: 14px 18px; margin-bottom: 16px; font-size: 13px; }}
        .verify-box strong {{ color: #bf6d00; }}
        .weight-tag, .temp-tag {{ display: inline-block; border-radius: 4px; padding: 2px 8px; margin: 2px 4px 2px 0; font-family: 'Courier New', monospace; font-size: 12px; }}
        .weight-tag {{ background: var(--card-bg); color: var(--accent); border: 1px solid var(--border); }}
        .temp-tag {{ background: #fff3e0; color: #e65100; border: 1px solid #ffcc80; }}

        .data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        .data-table th {{
            background: #f8f9fa;
            text-align: left;
            padding: 8px 10px;
            font-weight: 600;
            color: var(--text-muted);
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 2px solid var(--border);
            position: sticky;
            top: 0;
            z-index: 1;
        }}
        .data-table td {{ padding: 7px 10px; border-bottom: 1px solid #f1f3f5; color: var(--text); vertical-align: top; }}
        .data-table tr:hover td {{ background: #f8f9fa; }}
        .data-table tr:last-child td {{ border-bottom: none; }}
        .data-table th.sortable {{ cursor: pointer; user-select: none; white-space: nowrap; }}
        .data-table th.sortable::after {{ content: ' ⇅'; color: #adb5bd; font-weight: 400; }}
        .data-table th.sort-asc::after {{ content: ' ↑'; color: var(--accent); }}
        .data-table th.sort-desc::after {{ content: ' ↓'; color: var(--accent); }}

        .pos-cell {{ color: var(--positive); font-weight: 600; }}
        .neg-cell {{ color: var(--negative); font-weight: 600; }}
        .buy-cell {{ color: #1565C0; font-weight: 600; }}
        .sell-cell {{ color: #6A1B9A; font-weight: 600; }}
        .stock-text {{ font-family: 'Courier New', monospace; }}
        .code-btn {{
            background: var(--accent-light);
            color: var(--accent);
            border: 1px solid var(--accent);
            border-radius: 4px;
            padding: 2px 8px;
            font-size: 12px;
            font-family: 'Courier New', monospace;
            cursor: pointer;
            transition: all 0.15s;
            text-align: left;
        }}
        .code-btn:hover {{ background: var(--accent); color: #fff; }}
        .muted {{ color: var(--text-muted); }}

        .table-wrapper {{ overflow-x: auto; overflow-y: auto; max-height: 460px; }}
        .table-wrapper::-webkit-scrollbar {{ width: 6px; height: 6px; }}
        .table-wrapper::-webkit-scrollbar-thumb {{ background: #ccc; border-radius: 3px; }}

        .charts-2col, .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
        @media (max-width: 980px) {{ .charts-2col, .two-col {{ grid-template-columns: 1fr; }} }}

        .stats-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        .stats-table th {{ text-align: left; padding: 7px 12px; background: #f8f9fa; font-weight: 600; color: var(--text-muted); width: 46%; border-bottom: 1px solid var(--border); }}
        .stats-table td {{ padding: 7px 12px; border-bottom: 1px solid #f1f3f5; font-family: 'Courier New', monospace; }}
        .stats-table tr:last-child td {{ border-bottom: none; }}

        .footer {{ text-align: center; color: var(--text-muted); font-size: 12px; margin-top: 32px; padding-top: 16px; border-top: 1px solid var(--border); }}
        .no-data {{ color: var(--text-muted); font-size: 13px; padding: 24px; text-align: center; }}
        h2 {{ font-size: 16px; font-weight: 600; margin: 24px 0 14px; color: var(--text); border-bottom: 2px solid var(--accent); padding-bottom: 8px; }}

        .kline-overlay {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.6); z-index: 9998; }}
        .kline-panel {{ display: none; position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 95vw; height: 90vh; background: var(--card-bg); border-radius: 14px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); z-index: 9999; flex-direction: column; overflow: hidden; }}
        .kline-panel.active {{ display: flex; }}
        .kline-panel-header {{ display: flex; justify-content: space-between; align-items: center; padding: 14px 20px; background: #f8f9fa; border-bottom: 1px solid var(--border); font-weight: 600; font-size: 15px; flex-shrink: 0; }}
        .panel-close {{ background: none; border: none; font-size: 22px; cursor: pointer; color: var(--text-muted); line-height: 1; }}
        .panel-close:hover {{ color: var(--text); }}
        .kline-stock-pills {{ display: flex; flex-wrap: wrap; gap: 6px; padding: 12px 16px; background: #fff; border-bottom: 1px solid var(--border); flex-shrink: 0; max-height: 120px; overflow-y: auto; }}
        .stock-pill {{ padding: 4px 12px; border-radius: 16px; border: 1px solid var(--border); background: var(--accent-light); color: var(--accent); font-size: 12px; font-family: 'Courier New', monospace; cursor: pointer; transition: all 0.15s; }}
        .stock-pill:hover {{ background: var(--accent); color: #fff; }}
        .stock-pill.no-data {{ opacity: 0.4; cursor: default; }}
        .kline-charts-area {{ flex: 1; overflow: auto; padding: 8px; }}
        .kline-placeholder {{ display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-muted); font-size: 15px; }}
        .kline-chart {{ padding: 8px; }}
        .kline-header {{ display: flex; justify-content: space-between; align-items: center; padding: 8px 0 4px; }}
        .kline-code {{ font-size: 15px; font-weight: 700; font-family: 'Courier New', monospace; color: var(--text); }}
        .kline-close {{ background: none; border: 1px solid var(--border); border-radius: 6px; padding: 4px 12px; cursor: pointer; font-size: 13px; color: var(--text-muted); }}
        .kline-close:hover {{ background: #f1f3f5; }}

        .load-more-btn {{
            display: block;
            width: 100%;
            padding: 12px;
            margin-top: 8px;
            background: var(--accent-light);
            color: var(--accent);
            border: 1px dashed var(--accent);
            border-radius: 8px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
            transition: all 0.15s;
        }}
        .load-more-btn:hover {{
            background: var(--accent);
            color: #fff;
        }}
    </style>
</head>
<body>
<div class="container">
    {kline_section}

    <div class="report-header">
        <div class="report-title">回测报告</div>
        <div class="report-subtitle">WBR 量化交易系统 · T-1 信号 / T 日开盘执行 详细报告</div>
        <div class="report-meta">
            <span>信号周期: {html_escape(signal_period_str)}</span>
            <span>执行周期: {html_escape(trade_period_str or period_str)}</span>
            <span>调仓日数: {trade_days}</span>
            <span>初始资金: {_fmt_money(init_cash)}</span>
            <span>最终资产: {_fmt_money(final_asset)}</span>
            <span>生成时间: {html_escape(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</span>
        </div>
    </div>

    <div class="config-box">
        <strong>因子权重:</strong> {weights_html}<br>
        <strong>温度参数:</strong> {temps_html}<br>
        <strong>调仓配置:</strong> buy_n={buy_n}, sell_m={sell_m} &nbsp;|&nbsp;
        <strong>冻结天数:</strong> {freeze_days} 交易日<br>
        <strong>调仓规则:</strong> 信号={html_escape(signal_timing)} &nbsp;|&nbsp; 执行={html_escape(trade_timing)} &nbsp;|&nbsp; 价格字段={html_escape(price_field)}<br>
        <strong>复权口径:</strong> 信号={html_escape(signal_dividend_type)} &nbsp;→&nbsp; 执行={html_escape(execution_dividend_type)} &nbsp;|&nbsp;
        <strong>实际买入次数:</strong> {metrics.get('executed_buy_count', 0)} &nbsp;|&nbsp;
        <strong>实际卖出次数:</strong> {metrics.get('executed_sell_count', 0)} &nbsp;|&nbsp;
        <strong>完整 round-trip 数:</strong> {metrics.get('round_trip_count', 0)}
    </div>

    {verify_notice_html}

    <div class="card">
        <div class="card-title">核心指标</div>
        <div class="metrics-grid">{metric_cards}</div>
    </div>

    <div class="card">
        <div class="card-title">累计收益曲线</div>
        {equity_chart}
    </div>

    <div class="charts-2col">
        <div class="card">
            <div class="card-title">每日收益率分布</div>
            {dist_chart}
        </div>
        <div class="card">
            <div class="card-title">盈亏分布</div>
            {winloss_chart}
        </div>
    </div>

    <div class="two-col">
        <div class="card">
            <div class="card-title">交易统计口径</div>
            {trade_stats_html}
        </div>
        <div class="card">
            <div class="card-title">持仓天数统计（交易日）</div>
            {holding_stats_html}
        </div>
    </div>

    <div class="card">
        <div class="card-title">月度收益</div>
        <div class="table-wrapper">{monthly_table}</div>
    </div>

    <h2>交易记录明细</h2>
    <div class="card">
        <div class="table-wrapper">{trade_table}</div>
    </div>

    <h2>当前持仓明细 ({len(positions)} 只)</h2>
    <div class="card">
        <div class="table-wrapper">{holdings_table}</div>
    </div>

    <h2>已清仓持仓明细 ({len(cleared_positions)} 只)</h2>
    <div class="card">
        <div class="table-wrapper">{cleared_table}</div>
    </div>

    <h2>退市归零事件 ({len(delist_events)} 只)</h2>
    <div class="card">
        <div class="table-wrapper">{delist_table}</div>
    </div>

    <h2>每日资金快照</h2>
    <div class="card">
        <div class="table-wrapper">{daily_table}</div>
    </div>

    <div class="footer">
        WBR 量化交易系统 · 回测报告 · 实际成交 {len(trade_log)} 笔 · {trade_days} 个交易日
    </div>
</div>

<script>
/* ---- K 线懒加载：数据 gzip+base64 压缩存储，点击时解压渲染 ---- */
var _klineCache = null;
async function _loadKline() {{
    if (_klineCache) return _klineCache;
    if (typeof KLINE_B64 === 'undefined') return {{}};
    var bin = atob(KLINE_B64);
    var u8 = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    var ds = new DecompressionStream('gzip');
    var w = ds.writable.getWriter();
    w.write(u8); w.close();
    var r = ds.readable.getReader();
    var chunks = [];
    while (true) {{
        var res = await r.read();
        if (res.done) break;
        chunks.push(res.value);
    }}
    _klineCache = JSON.parse(await new Blob(chunks).text());
    KLINE_B64 = null;
    return _klineCache;
}}

async function showKline(code) {{
    if (typeof KLINE_B64 === 'undefined' && !_klineCache) return;
    var data = await _loadKline();
    var s = data[code];
    if (!s) return;

    var area = document.getElementById('klineRenderArea');
    if (!area) return;
    area.style.display = 'block';
    var ph = document.getElementById('klinePlaceholder');
    if (ph) ph.style.display = 'none';

    var traces = [{{
        type:'candlestick', x:s.d, open:s.o, high:s.h, low:s.l, close:s.c,
        name:'K线',
        increasing:{{line:{{color:'#26A69A'}},fillcolor:'#26A69A'}},
        decreasing:{{line:{{color:'#EF5350'}},fillcolor:'#EF5350'}},
        opacity:0.9
    }}];
    if (s.b && s.b.length) traces.push({{
        type:'scatter', mode:'markers', name:'买入',
        x:s.b.map(function(t){{return t.d}}), y:s.b.map(function(t){{return t.p}}),
        marker:{{symbol:'triangle-up',size:14,color:'#2E7D32',line:{{width:1,color:'white'}}}},
        hovertemplate:'买入<br>执行日: %{{x}}<br>价格: %{{y:.4f}}<extra></extra>'
    }});
    if (s.s && s.s.length) traces.push({{
        type:'scatter', mode:'markers', name:'卖出',
        x:s.s.map(function(t){{return t.d}}), y:s.s.map(function(t){{return t.p}}),
        marker:{{symbol:'triangle-down',size:14,color:'#C62828',line:{{width:1,color:'white'}}}},
        hovertemplate:'卖出<br>执行日: %{{x}}<br>价格: %{{y:.4f}}<extra></extra>'
    }});

    Plotly.newPlot(area, traces, {{
        title:{{text:'K线走势 · '+code+' '+(s.n||''), x:0.5, font:{{size:15}}}},
        height:500, showlegend:true,
        legend:{{orientation:'h',yanchor:'bottom',y:1.02,xanchor:'right',x:1}},
        hovermode:'x unified',
        xaxis:{{rangeslider:{{visible:false}}}}
    }});

    document.getElementById('klineOverlay').style.display = 'block';
    document.getElementById('klinePanel').classList.add('active');
}}

function closeKline() {{
    var overlay = document.getElementById('klineOverlay');
    var panel = document.getElementById('klinePanel');
    if (overlay) overlay.style.display = 'none';
    if (panel) panel.classList.remove('active');
    var area = document.getElementById('klineRenderArea');
    if (area) {{ area.style.display = 'none'; try {{ Plotly.purge(area); }} catch(e) {{}} }}
    var ph = document.getElementById('klinePlaceholder');
    if (ph) ph.style.display = 'flex';
}}

document.addEventListener('keydown', function(e) {{
    if (e.key === 'Escape') closeKline();
}});

/* ---- 表格排序 ---- */
function parseSortValue(value, type) {{
    if (type === 'number') {{
        var n = parseFloat(value);
        return Number.isNaN(n) ? Number.NEGATIVE_INFINITY : n;
    }}
    if (type === 'date') {{
        var t = Date.parse(value);
        return Number.isNaN(t) ? Number.NEGATIVE_INFINITY : t;
    }}
    return (value || '').toString().toLowerCase();
}}

function setupSortableTables() {{
    document.querySelectorAll('.sortable-table').forEach(function(table) {{
        var headers = table.querySelectorAll('thead th[data-sortable="true"]');
        headers.forEach(function(th, columnIndex) {{
            th.addEventListener('click', function() {{
                var tbody = table.querySelector('tbody');
                if (!tbody) return;
                var rows = Array.from(tbody.querySelectorAll('tr'));
                var currentDir = th.dataset.sortDir === 'asc' ? 'desc' : 'asc';
                headers.forEach(function(other) {{
                    other.dataset.sortDir = '';
                    other.classList.remove('sort-asc', 'sort-desc');
                }});
                th.dataset.sortDir = currentDir;
                th.classList.add(currentDir === 'asc' ? 'sort-asc' : 'sort-desc');
                th.classList.remove(currentDir === 'asc' ? 'sort-desc' : 'sort-asc');

                var sortType = th.dataset.sortType || 'text';
                rows.sort(function(a, b) {{
                    var aCell = a.children[columnIndex];
                    var bCell = b.children[columnIndex];
                    var aValue = parseSortValue(aCell.dataset.sort || aCell.innerText, sortType);
                    var bValue = parseSortValue(bCell.dataset.sort || bCell.innerText, sortType);
                    if (aValue < bValue) return currentDir === 'asc' ? -1 : 1;
                    if (aValue > bValue) return currentDir === 'asc' ? 1 : -1;
                    return 0;
                }});
                rows.forEach(function(row) {{ tbody.appendChild(row); }});
            }});
        }});
    }});
}}

/* ---- 大表格懒加载：默认只显示前 100 行 ---- */
function initLazyTables() {{
    var PAGE = 100;
    document.querySelectorAll('.data-table tbody').forEach(function(tbody) {{
        var rows = tbody.querySelectorAll('tr');
        if (rows.length <= PAGE) return;
        for (var i = PAGE; i < rows.length; i++) rows[i].style.display = 'none';
        var btn = document.createElement('button');
        btn.className = 'load-more-btn';
        btn.textContent = '展开全部 ' + rows.length + ' 行（当前显示前 ' + PAGE + ' 行）';
        btn.onclick = function() {{
            for (var i = PAGE; i < rows.length; i++) rows[i].style.display = '';
            btn.remove();
        }};
        var wrapper = tbody.closest('.table-wrapper') || tbody.parentNode.parentNode;
        wrapper.appendChild(btn);
    }});
}}

document.addEventListener('DOMContentLoaded', function() {{
    setupSortableTables();
    initLazyTables();
}});
</script>
</body>
</html>"""

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    with open(stable_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    testback_logger.info(f'详细回测报告已生成: {html_path}')
    testback_logger.info(f'单次回测报告已更新: {stable_html_path}')
    return stable_html_path
