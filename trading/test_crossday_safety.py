from datetime import date
from types import SimpleNamespace as NS

import numpy as np
import pytest


@pytest.mark.parametrize('asset,positions', [
    (None, []), (NS(cash=100, total_asset=100, market_value=0), None),
    (NS(cash=100, total_asset=1100, market_value=1000), []),
    (NS(cash=float('nan'), total_asset=100, market_value=0), []),
    (NS(cash=0, total_asset=0, market_value=0), []),
])
def test_account_snapshot_fails_closed(asset, positions):
    from trading.trader import Trader
    trader = object.__new__(Trader)
    trader.account = 'test'
    trader.client = NS(query_stock_asset=lambda a: asset, query_stock_positions=lambda a: positions)
    with pytest.raises(RuntimeError, match='账户快照不可信'):
        trader.query_account_snapshot(attempts=1)


def test_genuine_empty_funded_account_is_valid():
    from trading.trader import Trader
    trader = object.__new__(Trader)
    trader.account = 'test'
    asset = NS(cash=1000, total_asset=1000, market_value=0)
    trader.client = NS(query_stock_asset=lambda a: asset, query_stock_positions=lambda a: [])
    assert trader.query_account_snapshot(attempts=1) == (asset, [])


def test_missing_previous_session_rejected(tmp_path, monkeypatch):
    import core.runtime as runtime
    import trading.main as main
    path = tmp_path / 'runtime.npz'
    monkeypatch.setattr(runtime, 'latest_runtime_npz_path', lambda: path)
    monkeypatch.setattr(main, 'get_last_trading_day', lambda day: date(2026, 9, 4))
    np.savez(path, trade_dates=np.array(['2026-09-03'], dtype='datetime64[D]'))
    with pytest.raises(RuntimeError, match='上一交易日'):
        main._require_previous_session(date(2026, 9, 7))
    np.savez(path, trade_dates=np.array(['2026-09-04'], dtype='datetime64[D]'))
    main._require_previous_session(date(2026, 9, 7))


def test_lark_markdown_table_numeric_cells_are_strings():
    from trading.lark.sender import make_v2_table
    table = make_v2_table(columns=[{'name': 'count'}, {'name': 'shares'}, {'name': 'empty'}],
                          rows=[{'count': 1, 'shares': 200, 'empty': None}])
    assert table['rows'] == [{'count': '1', 'shares': '200', 'empty': ''}]


def test_readonly_trader_rejects_order_and_cancel():
    from trading.trader import Trader
    trader = object.__new__(Trader)
    trader.enable_orders = False
    with pytest.raises(RuntimeError, match='只读'):
        trader.order(23, '600000.SH', 100, 10)
    with pytest.raises(RuntimeError, match='只读'):
        trader.cancel_order(1)


def test_old_day_order_monitor_cannot_submit_next_day(monkeypatch):
    from datetime import datetime
    import trading.executor as module
    ex = module.RebalanceExecutor(object())
    monitor = module.OrderMonitor(ex, [], {}, [], {}, date(2026, 9, 4), date(2026, 9, 4))
    monkeypatch.setattr(module, 'is_current_trading', lambda now: True)
    assert not monitor._can_submit_now(datetime(2026, 9, 7, 9, 30).timestamp())
