import json
from datetime import date

import pytest

from core.sim.account import StockAccountMocker
from testback.metrics import compute_strategy_metrics
from testback.reportor import generate_single_report
from testback.reportor.report import _make_trade_table


def _account_with_visible_costs() -> StockAccountMocker:
    return StockAccountMocker(
        cash=100_000.0,
        commission=0.001,
        min_commission=0.0,
        stamp_tax=0.002,
        transfer_fee=0.003,
        slippage=0.004,
    )


def test_trade_log_splits_cost_components_and_keeps_legacy_total():
    account = _account_with_visible_costs()

    assert account.buy_stock('000001', 100, 10.0, date(2024, 1, 2))
    account.sell_stock('000001', 100, 12.0, date(2024, 1, 3))
    assert account.buy_stock('000002', 100, 5.0, date(2024, 1, 3))
    account.write_off_stock('000002', date(2024, 1, 4))

    buy, sell, _, write_off = account.get_trade_log()
    assert buy['broker_commission'] == pytest.approx(1.0)
    assert buy['transfer_fee'] == pytest.approx(3.0)
    assert buy['stamp_tax'] == 0.0
    assert buy['slippage'] == pytest.approx(4.0)
    assert buy['total_fee'] == pytest.approx(8.0)
    assert buy['commission'] == buy['total_fee']

    assert sell['broker_commission'] == pytest.approx(1.2)
    assert sell['transfer_fee'] == pytest.approx(3.6)
    assert sell['stamp_tax'] == pytest.approx(2.4)
    assert sell['slippage'] == pytest.approx(4.8)
    assert sell['total_fee'] == pytest.approx(12.0)
    assert sell['commission'] == sell['total_fee']

    for field in (
        'commission',
        'broker_commission',
        'transfer_fee',
        'stamp_tax',
        'slippage',
        'total_fee',
    ):
        assert write_off[field] == 0.0


def test_metrics_report_each_cost_and_preserve_total_commission_alias():
    account = _account_with_visible_costs()
    assert account.buy_stock('000001', 100, 10.0, date(2024, 1, 2))
    account.sell_stock('000001', 100, 12.0, date(2024, 1, 3))
    assert account.buy_stock('000002', 100, 5.0, date(2024, 1, 3))
    account.write_off_stock('000002', date(2024, 1, 4))

    metrics = compute_strategy_metrics(
        [0.0, 1.0, 0.5],
        ['2024-01-02', '2024-01-03', '2024-01-04'],
        account.get_trade_log(),
    )

    assert metrics['total_broker_commission'] == pytest.approx(2.7)
    assert metrics['total_transfer_fee'] == pytest.approx(8.1)
    assert metrics['total_stamp_tax'] == pytest.approx(2.4)
    assert metrics['total_slippage'] == pytest.approx(10.8)
    assert metrics['total_fees'] == pytest.approx(24.0)
    assert metrics['total_commission'] == metrics['total_fees']


def test_report_json_and_trade_table_expose_slippage_and_total_cost(tmp_path):
    metrics = {
        'total_broker_commission': 2.7,
        'total_transfer_fee': 8.1,
        'total_stamp_tax': 2.4,
        'total_slippage': 10.8,
        'total_fees': 24.0,
        'total_commission': 24.0,
    }
    report_path = generate_single_report(
        {
            'individual_config': {'weights': {}, 'buy_n': 1, 'sell_m': 1},
            'cumulative_returns': [0.0],
            'trade_dates': ['2024-01-02'],
            'trade_log': [],
            'metrics': metrics,
            'period': {'start': '2024-01-02', 'end': '2024-01-02'},
        },
        tmp_path,
    )
    html = report_path.read_text(encoding='utf-8')
    marker = '<script id="report-data" type="application/json">'
    payload = json.loads(html.split(marker, 1)[1].split('</script>', 1)[0])

    assert payload['summary']['total_fees'] == 24.0
    assert payload['summary']['total_commission'] == 24.0
    assert payload['summary']['total_broker_commission'] == 2.7
    assert payload['summary']['total_transfer_fee'] == 8.1
    assert payload['summary']['total_stamp_tax'] == 2.4
    assert payload['summary']['total_slippage'] == 10.8
    for label in ('总交易成本', '总滑点', '券商佣金', '印花税', '过户费'):
        assert label in html
    assert '总手续费' not in html

    table = _make_trade_table(
        [{
            'code': '000001',
            'action': 'buy',
            'date': '2024-01-02',
            'price': 10.0,
            'volume': 100,
            'amount': 1_000.0,
            'broker_commission': 1.0,
            'transfer_fee': 3.0,
            'stamp_tax': 0.0,
            'slippage': 4.0,
            'total_fee': 8.0,
            'commission': 8.0,
        }],
        {},
    )
    labels = [header['label'] for header in table['headers']]
    assert '滑点(¥)' in labels
    assert '总费用(¥)' in labels
    row = table['rows'][0]
    assert float(row[labels.index('滑点(¥)')]['sort']) == 4.0
    assert float(row[labels.index('总费用(¥)')]['sort']) == 8.0

    legacy_table = _make_trade_table(
        [{
            'code': '000001',
            'action': 'buy',
            'date': '2024-01-02',
            'price': 10.0,
            'volume': 100,
            'amount': 1_000.0,
            'commission': 9.0,
        }],
        {},
    )
    assert (
        float(legacy_table['rows'][0][labels.index('总费用(¥)')]['sort'])
        == 9.0
    )
