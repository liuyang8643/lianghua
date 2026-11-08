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
    from trading.account_snapshot import validate_account_snapshot
    with pytest.raises(ValueError):validate_account_snapshot(asset,positions)


def test_genuine_empty_funded_account_is_valid():
    from trading.account_snapshot import validate_account_snapshot
    asset=NS(cash=1000,total_asset=1000,market_value=0)
    assert validate_account_snapshot(asset,[])==(asset,[])


def test_missing_previous_session_rejected(tmp_path,monkeypatch):
    from trading import startup_data as module
    path=tmp_path/'runtime.npz'
    monkeypatch.setattr(module,'latest_runtime_npz_path',lambda:path)
    monkeypatch.setattr(module,'get_trading_date_span',lambda *args:[date(2026,9,4)])
    arrays={name:np.ones((1,1)) for name in module._2D_FIELDS}
    arrays.update({name:np.ones(1) for name in module._1D_FIELDS})
    arrays['stock_codes']=np.array(['600001.SH'])
    np.savez(path,trade_dates=np.array(['2026-09-03'],dtype='datetime64[D]'),**arrays)
    with pytest.raises(ValueError,match='缺少历史交易日'):module.validate_startup_runtime(date(2026,9,4))
    np.savez(path,trade_dates=np.array(['2026-09-04'],dtype='datetime64[D]'),**arrays)
    assert module.validate_startup_runtime(date(2026,9,4))==path


def test_lark_markdown_table_numeric_cells_are_strings():
    from trading.lark.sender import make_v2_table
    table = make_v2_table(columns=[{'name': 'count'}, {'name': 'shares'}, {'name': 'empty'}],
                          rows=[{'count': 1, 'shares': 200, 'empty': None}])
    assert table['rows'] == [{'count': '1', 'shares': '200', 'empty': ''}]


def test_local_file_account_always_rejects_order_and_cancel():
    from trading.file_account import FileAccount
    account=FileAccount('test')
    with pytest.raises(RuntimeError,match='cannot submit'):account.order(23,'600001.SH',100,10)
    with pytest.raises(RuntimeError,match='cannot cancel'):account.cancel_order(1)


def test_old_day_order_monitor_cannot_submit_next_day(monkeypatch):
    from datetime import datetime
    import trading.execution_engine as module
    ex = module.RebalanceExecutor(object())
    monitor = module.OrderMonitor(ex, [], {}, [], {}, date(2026, 9, 4), date(2026, 9, 4))
    monkeypatch.setattr(module.RebalanceExecutor, '_is_current_trading', lambda self, now: True)
    assert not monitor._can_submit_now(datetime(2026, 9, 7, 9, 30).timestamp())
