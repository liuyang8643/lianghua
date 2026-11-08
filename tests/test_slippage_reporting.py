import json

import numpy as np
import pytest

from env.backtest import RolloutTrace
from env.contracts import AccountState, OrderPlan, StepResult
from env.fees import FeeSchedule
from env.simulator import DaySimulator
from testback.backtest import _core_report_metrics
from testback.reportor import generate_single_report
from testback.reportor.report import _make_trade_table


VISIBLE_FEES = FeeSchedule(
    commission_rate=0.001,
    minimum_commission=0.0,
    stamp_tax_rate=0.002,
    transfer_fee_rate=0.003,
    slippage_rate=0.004,
)


def _run_visible_cost_fills() -> tuple[StepResult, StepResult, StepResult]:
    simulator = DaySimulator(VISIBLE_FEES)
    first_buy = simulator.step(
        AccountState(cash=100_000.0, nav=100_000.0, peak_nav=100_000.0),
        OrderPlan(
            decision_date="2024-01-02",
            buy_orders={"000001.SZ": 100},
        ),
        {"000001.SZ": 10.0},
        {"000001.SZ": 10.0},
        close_prices={"000001.SZ": 10.0},
        next_preclose_prices={"000001.SZ": 10.0},
    )
    sell = simulator.step(
        first_buy.account_state,
        OrderPlan(
            decision_date="2024-01-03",
            sell_orders=(("000001.SZ", 100),),
        ),
        {"000001.SZ": 12.0},
        {},
    )
    second_buy = simulator.step(
        sell.account_state,
        OrderPlan(
            decision_date="2024-01-04",
            buy_orders={"000002.SZ": 100},
        ),
        {"000002.SZ": 5.0},
        {"000002.SZ": 5.0},
        close_prices={"000002.SZ": 5.0},
        next_preclose_prices={"000002.SZ": 5.0},
    )
    return first_buy, sell, second_buy


def _canonical_trade_log(*results: StepResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        breakdowns = result.diagnostics["fee_breakdown"]
        for fill, costs in zip(result.fills, breakdowns, strict=True):
            rows.append(
                {
                    "action": fill.side,
                    "broker_commission": costs["commission"],
                    "transfer_fee": costs["transfer_fee"],
                    "stamp_tax": costs["stamp_tax"],
                    "slippage": costs["slippage"],
                    "total_fee": costs["total"],
                }
            )
    return rows


def test_day_simulator_splits_cost_components_and_fill_keeps_total():
    buy_result, sell_result, _ = _run_visible_cost_fills()
    buy = buy_result.diagnostics["fee_breakdown"][0]
    sell = sell_result.diagnostics["fee_breakdown"][0]

    assert buy["commission"] == pytest.approx(1.0)
    assert buy["transfer_fee"] == pytest.approx(3.0)
    assert buy["stamp_tax"] == 0.0
    assert buy["slippage"] == pytest.approx(4.0)
    assert buy["total"] == pytest.approx(8.0)
    assert buy_result.fills[0].fee == buy["total"]

    assert sell["commission"] == pytest.approx(1.2)
    assert sell["transfer_fee"] == pytest.approx(3.6)
    assert sell["stamp_tax"] == pytest.approx(2.4)
    assert sell["slippage"] == pytest.approx(4.8)
    assert sell["total"] == pytest.approx(12.0)
    assert sell_result.fills[0].fee == sell["total"]


def test_canonical_report_metrics_aggregate_each_cost_once():
    results = _run_visible_cost_fills()
    trace = RolloutTrace(
        decision_dates=("2024-01-02", "2024-01-03", "2024-01-04"),
        next_decision_dates=("2024-01-03", "2024-01-04", "2024-01-05"),
        rewards=np.zeros(3, dtype=np.float64),
        portfolio_returns=np.zeros(3, dtype=np.float64),
        nav=np.full(4, 100_000.0, dtype=np.float64),
        cash=np.full(3, 50_000.0, dtype=np.float64),
        exposure=np.full(3, 0.5, dtype=np.float64),
        full_investment_contract=np.ones(3, dtype=np.bool_),
        residual_cash_reasons=("capacity_exhausted",) * 3,
        order_plans=({},) * 3,
        fills=tuple(tuple(result.fills) for result in results),
        fee_breakdowns=tuple(
            tuple(result.diagnostics["fee_breakdown"]) for result in results
        ),
        account_events=(),
        actions=np.zeros((3, 1), dtype=np.float32),
        day_configs=({},) * 3,
    )

    metrics = _core_report_metrics(
        {
            'trace': trace,
            'daily_exposures': [0.5, 0.5, 0.5],
            'trade_log': _canonical_trade_log(*results),
            'executed_buy_count': 2,
            'executed_sell_count': 1,
        }
    )

    assert metrics['total_broker_commission'] == pytest.approx(2.7)
    assert metrics['total_transfer_fee'] == pytest.approx(8.1)
    assert metrics['total_stamp_tax'] == pytest.approx(2.4)
    assert metrics['total_slippage'] == pytest.approx(10.8)
    assert metrics['total_fees'] == pytest.approx(24.0)
    assert 'total_commission' not in metrics


def test_report_json_and_trade_table_expose_slippage_and_total_cost(tmp_path):
    metrics = {
        'total_broker_commission': 2.7,
        'total_transfer_fee': 8.1,
        'total_stamp_tax': 2.4,
        'total_slippage': 10.8,
        'total_fees': 24.0,
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
    assert 'total_commission' not in payload['summary']
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
        }],
        {},
    )
    labels = [header['label'] for header in table['headers']]
    assert '滑点(¥)' in labels
    assert '总费用(¥)' in labels
    row = table['rows'][0]
    assert float(row[labels.index('滑点(¥)')]['sort']) == 4.0
    assert float(row[labels.index('总费用(¥)')]['sort']) == 8.0
