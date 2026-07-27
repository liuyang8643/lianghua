import json
from datetime import datetime

from testback.reportor import generate_single_report
from testback.reportor.report import _build_trade_episodes, _make_trade_table


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

    marker = '<script id="report-data" type="application/json">'
    payload_text = html.split(marker, 1)[1].split('</script>', 1)[0]
    payload = json.loads(payload_text)
    assert payload['summary']['excess_return_pct'] == 0.9
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
