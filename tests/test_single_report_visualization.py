import json
from datetime import datetime

import numpy as np
import pytest

from testback.reportor import generate_single_report
from testback.reportor.report import (
    _build_trade_episodes,
    _collect_kline_payload,
    _make_daily_table,
    _make_delist_events_table,
    _make_trade_table,
)


def test_single_report_renders_research_workspace(tmp_path):
    dates = ['2024-01-02', '2024-01-03', '2024-02-01']
    snapshots = [
        {
            'date': date,
            'signal_date': date,
            'trade_date': date,
            'price_field': 'open',
            'cash': 300_000.0,
            'market_value': 700_000.0 + idx * 10_000,
            'total_asset': 1_000_000.0 + idx * 10_000,
            'daily_return_pct': [1.0, -0.5, 1.2][idx],
            'cumulative_return_pct': [1.0, 0.5, 1.7][idx],
            'exposure': 0.7,
            'rebalance_funds_ratio': [0.4, 0.0, 0.25][idx],
            'buy_n_list': ['000001'],
            'executed_buy_list': ['000001'] if idx == 0 else [],
            'executed_sell_list': [],
            'entered_stocks': ['000001'] if idx == 0 else [],
            'exited_stocks': [],
        }
        for idx, date in enumerate(dates)
    ]
    report_data = {
        'individual_config': {'weights': {'DemoFactor': 1.0}, 'buy_n': 1, 'sell_m': 2},
        'total_return': 1.7,
        'daily_returns': [1.0, -0.5, 1.2],
        'cumulative_returns': [1.0, 0.5, 1.7],
        'trade_dates': dates,
        'trade_log': [],
        'daily_snapshots': snapshots,
        'positions': [],
        'cleared_positions': [],
        'delist_events': [],
        'stock_name_map': {'000001': '平安银行'},
        'holding_stats': {},
        'executed_buy_count': 1,
        'executed_sell_count': 0,
        'round_trip_count': 0,
        'final_asset': 1_017_000.0,
        'metrics': {
            'annualized': 12.3,
            'max_drawdown': -0.5,
            'max_drawdown_start': dates[0],
            'max_drawdown_end': dates[1],
            'sharpe_ratio': 1.1,
            'calmar_ratio': 2.0,
            'win_rate': 0.0,
            'average_exposure': 0.7,
        },
        'per_year_metrics': [],
        'hs300_returns': [0.5, 0.2, 0.8],
        'factor_missing_counts': {'DemoFactor': [2, 3, 1], 'K线缺失': [1, 0, 2]},
        'report_metadata': {'stock_pool_size': 10},
        'init_cash': 1_000_000.0,
        'rebalance_rule': {'signal_timing': 'T-1', 'trade_timing': 'T open', 'price_field': 'open'},
        'period': {'start': dates[0], 'end': dates[-1], 'trade_start': dates[0], 'trade_end': dates[-1]},
    }

    html_path = generate_single_report(report_data, tmp_path)
    html = html_path.read_text(encoding='utf-8')

    assert 'id="drawdown-chart"' not in html
    assert 'id="exposure-chart"' not in html
    assert 'id="equity-chart"' in html
    assert 'id="factor-valid-chart"' in html
    assert 'id="klinePanel"' in html
    assert 'id="monthly-heatmap"' in html
    assert 'data-tab="trades"' in html
    assert 'renderPerformanceCharts();' in html
    assert 'min-width:320px' in html
    assert 'T-1 信号 · T open 执行' in html

    marker = '<script id="report-data" type="application/json">'
    payload_text = html.split(marker, 1)[1].split('</script>', 1)[0]
    payload = json.loads(payload_text)
    assert payload['summary']['excess_return_pct'] == 0.9
    assert payload['summary']['avg_daily_buys'] == 0.33
    assert payload['summary']['avg_daily_sells'] == 0.0
    assert payload['charts']['equity']['drawdown_pct'] == [0.0, -0.49505, 0.0]
    assert payload['charts']['equity']['rebalance_funds_pct'] == [40.0, 0.0, 25.0]
    assert payload['charts']['factor_valid']['series']['DemoFactor'] == [8, 7, 9]
    assert payload['charts']['factor_valid']['series']['K线有效'] == [9, 10, 8]
    assert payload['charts']['monthly'][0]['month'] == '2024-01'


def test_trade_episode_window_and_kline_button():
    trades = [
        {'code': '000001', 'action': 'buy', 'trade_date': '2024-01-10', 'price': 10, 'volume': 100},
        {'code': '000001', 'action': 'buy', 'trade_date': '2024-02-01', 'price': 11, 'volume': 100},
        {'code': '000001', 'action': 'sell', 'trade_date': '2024-03-01', 'price': 12, 'volume': 100},
        {'code': '000001', 'action': 'sell', 'trade_date': '2024-04-01', 'price': 13, 'volume': 100},
    ]
    for trade in trades:
        trade.update(
            {
                'total_fee': 0.0,
                'broker_commission': 0.0,
                'transfer_fee': 0.0,
                'stamp_tax': 0.0,
                'slippage': 0.0,
            }
        )

    resolved = _build_trade_episodes(trades, '2024-12-31')['000001']
    episode = resolved['episodes'][0]
    assert episode['start'] == '2024-01-10'
    assert episode['end'] == '2024-04-01'
    assert all(event['episode'] == 0 for event in resolved['events'])
    assert (
        datetime.strptime(episode['start'], '%Y-%m-%d')
        - datetime.strptime(episode['window_start'], '%Y-%m-%d')
    ).days == 183
    assert (
        datetime.strptime(episode['window_end'], '%Y-%m-%d')
        - datetime.strptime(episode['end'], '%Y-%m-%d')
    ).days == 183

    table = _make_trade_table(trades, {'000001': '平安银行'})
    assert 'data-kline-code="000001"' in table['rows'][0][2]['html']
    assert 'data-kline-event="0"' in table['rows'][0][2]['html']


def test_daily_table_separates_execution_from_next_open_settlement():
    table = _make_daily_table(
        [
            {
                'signal_date': '2024-01-02',
                'trade_date': '2024-01-02',
                'settlement_date': '2024-01-03',
                'cash': 100.0,
                'market_value': 900.0,
                'total_asset': 1_000.0,
            }
        ],
        {},
    )
    labels = [header['label'] for header in table['headers']]
    row = table['rows'][0]
    assert row[labels.index('信号日')]['sort'] == '2024-01-02'
    assert row[labels.index('执行日')]['sort'] == '2024-01-02'
    assert row[labels.index('估值/结算日')]['sort'] == '2024-01-03'


def test_delist_table_uses_canonical_account_event_fields():
    table = _make_delist_events_table(
        [{
            'type': 'delist_write_off',
            'code': '000001.SZ',
            'effective_date': '2024-01-03',
            'quantity': 100,
            'average_cost': 10.0,
            'proceeds': 0.0,
        }],
        {'000001.SZ': '退市样本'},
    )
    labels = [header['label'] for header in table['headers']]
    row = table['rows'][0]

    assert row[labels.index('归零执行日')]['sort'] == '2024-01-03'
    assert row[labels.index('数量(股)')]['sort'] == '100'
    assert row[labels.index('持仓成本(¥)')]['sort'] == '1000.0'
    assert row[labels.index('归零损失(¥)')]['sort'] == '-1000.0'
    assert row[labels.index('损失率')]['sort'] == '-100.0'


def test_canonical_report_marks_uncomputed_analytics_as_unavailable(tmp_path):
    execution_dates = ['2024-01-02', '2024-01-03']
    nav_dates = ['2024-01-03', '2024-01-04']
    report_data = {
        'individual_config': {
            'weights': {'DemoFactor': 1.0},
            'buy_n': 1,
            'sell_m': 2,
        },
        'total_return': 1.0,
        'daily_returns': [0.4, 0.6],
        'cumulative_returns': [0.4, 1.0],
        'trade_dates': execution_dates,
        'nav_dates': nav_dates,
        'trade_log': [],
        'daily_snapshots': [],
        'positions': [],
        'cleared_positions': [],
        'delist_events': [],
        'stock_name_map': {},
        'holding_stats': {},
        'executed_buy_count': 2,
        'executed_sell_count': 1,
        'round_trip_count': None,
        'cleared_positions_count': None,
        'position_lot_analytics_available': False,
        'holding_period_available': False,
        'benchmark_available': False,
        'per_year_metrics_available': False,
        'final_asset': 1_010_000.0,
        'metrics': {
            'annualized': 12.0,
            'max_drawdown': -1.0,
            'sharpe_ratio': 1.0,
            'calmar_ratio': 12.0,
            'total_trades': 3,
            'total_fees': 123.45,
        },
        'per_year_metrics': [],
        'hs300_returns': [],
        'report_metadata': {'stock_pool_size': 10},
        'init_cash': 1_000_000.0,
        'rebalance_rule': {
            'signal_timing': 'T-open',
            'trade_timing': 'T-open',
            'price_field': 'open',
        },
        'period': {
            'start': nav_dates[0],
            'end': nav_dates[-1],
            'signal_start': execution_dates[0],
            'signal_end': execution_dates[-1],
            'trade_start': execution_dates[0],
            'trade_end': execution_dates[-1],
        },
    }

    html_path = generate_single_report(report_data, tmp_path)
    html = html_path.read_text(encoding='utf-8')
    marker = '<script id="report-data" type="application/json">'
    payload_text = html.split(marker, 1)[1].split('</script>', 1)[0]
    payload = json.loads(payload_text)

    summary = payload['summary']
    visible_html = html.split(marker, 1)[0]
    assert summary['benchmark_available'] is False
    assert summary['benchmark_return_pct'] is None
    assert summary['excess_return_pct'] is None
    assert summary['position_lot_analytics_available'] is False
    assert summary['round_trips'] is None
    assert summary['wins'] is None
    assert summary['losses'] is None
    assert '沪深300 N/A（未计算）' in html
    assert 'N/A（未计算分年度指标）' in html
    assert '当前 canonical env 未提供逐持仓平仓配对，相关统计不展示' in html
    assert '相对沪深300' not in visible_html
    assert '完整 round-trip' not in visible_html
    assert '清仓胜率' not in visible_html
    assert '总交易成本' in visible_html
    assert '实际成交' in visible_html
    assert 'N/A（未提供逐持仓平仓配对）' in html


def test_kline_payload_reuses_sealed_runtime_without_per_stock_file_reads():
    dates = np.arange(
        np.datetime64('2024-01-01'),
        np.datetime64('2024-01-11'),
        dtype='datetime64[D]',
    )
    close = np.arange(10.0, 20.0, dtype=np.float64)[:, None]
    runtime_kline = {
        'trade_dates': dates,
        'stock_codes': np.asarray(['000001']),
        'open': close - 0.2,
        'high': close + 0.5,
        'low': close - 0.5,
        'close': close,
        'amount': np.full((10, 1), 1_000_000.0),
    }
    trades = [
        {
            'code': '000001',
            'action': 'buy',
            'trade_date': '2024-01-03',
            'price': 12.0,
            'volume': 100,
        },
        {
            'code': '000001',
            'action': 'sell',
            'trade_date': '2024-01-08',
            'price': 17.0,
            'volume': 100,
        },
    ]

    payload = _collect_kline_payload(
        trades,
        {'000001': '平安银行'},
        '2024-01-10',
        runtime_kline,
    )

    assert payload['000001']['d'] == [str(value) for value in dates]
    assert payload['000001']['c'] == list(np.arange(10.0, 20.0))
    assert payload['000001']['events'][0]['episode'] == 0


def test_kline_payload_rejects_trade_log_without_sealed_runtime():
    with pytest.raises(ValueError, match='sealed runtime snapshot'):
        _collect_kline_payload(
            [
                {
                    'code': '000001',
                    'action': 'buy',
                    'trade_date': '2024-01-03',
                    'price': 12.0,
                    'volume': 100,
                }
            ],
            {},
            '2024-01-10',
            None,
        )
