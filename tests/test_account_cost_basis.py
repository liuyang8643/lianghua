from datetime import date

import pytest

from core.rebalance import compute_rebalance_plan
from core.sim.account import (
    StockAccountMocker,
    calculate_buy_total_cost,
)
from testback.metrics import compute_strategy_metrics


def test_rebalance_reserves_minimum_commission_for_each_small_buy():
    first = "600001.SH"
    second = "600002.SH"
    one_order_cost = calculate_buy_total_cost(100.0)
    initial_cash = one_order_cost * 2 - 0.01

    _, buy_orders, skip_reasons = compute_rebalance_plan(
        positions={},
        sellable_volumes={},
        pos_vals={},
        cash=initial_cash,
        buy_n_stocks=[first, second],
        tradable_buy_stocks=[first, second],
        sellable_ok=set(),
        prices={first: 1.0, second: 1.0},
        limit_prices={first: 1.0, second: 1.0},
        base_target=100.0,
        slippage_bps=10.0,
    )

    assert buy_orders == {first: 100}
    assert skip_reasons[second] == "冻结资金不足"

    account = StockAccountMocker(cash=initial_cash)
    assert all(
        account.buy_stock(code, volume, 1.0, date(2024, 1, 2))
        for code, volume in buy_orders.items()
    )
    assert account.current_cash == pytest.approx(one_order_cost - 0.01)
    assert account.current_cash < one_order_cost


def test_partial_sales_allocate_buy_fees_and_full_clear_keeps_round_trip():
    code = "600001.SH"
    account = StockAccountMocker(
        cash=10_000.0,
        commission=0.01,
        min_commission=2.0,
        stamp_tax=0.03,
        transfer_fee=0.005,
        slippage=0.02,
    )

    assert account.buy_stock(code, 100, 10.0, date(2024, 1, 2))
    assert account.buy_stock(code, 100, 20.0, date(2024, 1, 3))

    position = account.positions[code]
    assert position["volume"] == 200
    assert position["cost"] == pytest.approx(3_105.0)
    assert position["avg_price"] == pytest.approx(15.525)
    assert account.current_cash == pytest.approx(6_895.0)

    account.sell_stock(code, 50, 30.0, date(2024, 1, 4))

    first_sell = account.trade_log[-1]
    assert first_sell["cost"] == pytest.approx(776.25)
    assert first_sell["income"] == pytest.approx(626.25)
    assert account.positions[code]["volume"] == 150
    assert account.positions[code]["cost"] == pytest.approx(2_328.75)
    assert account.positions[code]["avg_price"] == pytest.approx(15.525)
    assert account.current_cash == pytest.approx(8_297.5)

    account.sell_stock(code, 150, 10.0, date(2024, 1, 5))

    second_sell = account.trade_log[-1]
    assert second_sell["cost"] == pytest.approx(2_328.75)
    assert second_sell["income"] == pytest.approx(-926.25)
    assert account.current_cash == pytest.approx(9_700.0)
    assert account.calc_assets({})["total_asset"] == pytest.approx(9_700.0)
    assert code not in account.positions

    cleared = account.cleared_positions[-1]
    assert cleared["income"] == pytest.approx(-300.0)
    assert cleared["pos"]["volume"] == 200
    assert cleared["pos"]["cost"] == pytest.approx(3_105.0)
    assert cleared["pos"]["avg_price"] == pytest.approx(15.525)
    assert cleared["pos"]["commission"] == pytest.approx(300.0)
    assert sum(
        trade["income"]
        for trade in account.trade_log
        if trade["action"] == "sell"
    ) == pytest.approx(cleared["income"])

    metrics = compute_strategy_metrics(
        [0.0, 0.0, 0.0, -3.0],
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
        ],
        account.get_trade_log(),
    )
    assert metrics["wins"] == 1
    assert metrics["losses"] == 1
    assert metrics["avg_profit"] == pytest.approx(-150.0)
