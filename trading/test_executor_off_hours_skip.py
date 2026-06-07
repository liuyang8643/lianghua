"""闭市 --skip 时跳过真实 order_stock（避免 QMT 同步阻塞）。"""
from types import SimpleNamespace
from datetime import date

from trading.executor import RebalanceExecutor


class _NoOrderTrader:
    def query_asset(self):
        return SimpleNamespace(current_balance=0, cash=0, frozen_cash=0)


def test_off_hours_execute_skips_order_stock():
    ex = RebalanceExecutor(_NoOrderTrader())
    pending = {
        'signal_date': date(2026, 6, 4),
        'trade_date': date(2026, 6, 4),
        'sell_orders': [('002836.SZ', 100)],
        'buy_allocations': {'603151.SH': 100},
        'buy_n_stocks': ['603151.SH'],
        'prices': {'603151.SH': 10.0},
        'limit_prices': {},
    }
    out = ex.execute(pending, off_hours_fast=True)
    assert out == []
