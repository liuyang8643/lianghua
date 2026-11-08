from datetime import date, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from trading import startup_data as startup


PREVIOUS = date(2026, 9, 10)


def write_runtime(path, *, end=PREVIOUS, drop=None, empty_close=False):
    dates = np.array(startup.get_trading_date_span(date(2026, 6, 1), end),
                     dtype='datetime64[D]')
    arrays = {'trade_dates': dates, 'stock_codes': np.array(['000001.SZ', '600000.SH'])}
    arrays.update({key: np.ones((len(dates), 2)) for key in startup._2D_FIELDS})
    arrays.update({key: np.ones(2) for key in startup._1D_FIELDS})
    # 未上市/停牌/未披露财务的空值不应误判。
    arrays['close'][:, 1] = np.nan
    arrays['eps'][:] = np.nan
    if empty_close:
        arrays['close'][-1] = np.nan
    if drop:
        del arrays[drop]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


@pytest.fixture
def runtime_path(tmp_path, monkeypatch):
    path = tmp_path / 'runtime.npz'
    monkeypatch.setattr(startup, 'latest_runtime_npz_path', lambda: path)
    return path


def store():
    return SimpleNamespace(_now=lambda: datetime(2026, 9, 11, 1, 35))


def test_ready_runtime_skips_download(runtime_path, monkeypatch):
    write_runtime(runtime_path)
    monkeypatch.setattr('trading.post_close.run_update_all',
                        lambda *a, **k: pytest.fail('完整数据不得重复下载'))
    assert startup.ensure_startup_data(store()) is True


@pytest.mark.parametrize('damage', ['missing', 'stale', 'corrupt', 'field', 'empty'])
def test_startup_repairs_and_rechecks(runtime_path, monkeypatch, damage):
    if damage == 'stale':
        write_runtime(runtime_path, end=date(2026, 6, 2))
    elif damage == 'corrupt':
        runtime_path.write_bytes(b'bad zip')
    elif damage == 'field':
        write_runtime(runtime_path, drop='total_share')
    elif damage == 'empty':
        write_runtime(runtime_path, empty_close=True)
    calls = []

    def update(store, *, history_end, history_days):
        calls.append((history_end, history_days))
        write_runtime(runtime_path)
        return True

    monkeypatch.setattr('trading.post_close.run_update_all', update)
    assert startup.ensure_startup_data(store()) is True
    assert calls[0][0] == PREVIOUS
    if damage == 'stale':
        assert calls[0][1] >= (PREVIOUS - date(2026, 6, 2)).days


def test_download_failure_does_not_mark_ready(runtime_path, monkeypatch):
    monkeypatch.setattr('trading.post_close.run_update_all', lambda *a, **k: False)
    assert startup.ensure_startup_data(store()) is False


def test_download_success_requires_valid_runtime(runtime_path, monkeypatch):
    write_runtime(runtime_path, end=PREVIOUS - timedelta(days=1))
    monkeypatch.setattr('trading.post_close.run_update_all', lambda *a, **k: True)
    with pytest.raises(ValueError, match='缺少历史交易日'):
        startup.ensure_startup_data(store())


def test_scheduler_checks_before_open_and_retries_then_resets(monkeypatch):
    from trading.scheduler import TradingScheduler
    now = [datetime(2026, 9, 11, 1, 35)]
    calls = []

    def check(store):
        calls.append(('check', now[0]))
        return len(calls) > 1

    sch = TradingScheduler(object(), time_provider=lambda: now[0],
                           fast_forward=True, fast_forward_through_day=False,
                           ensure_data=check,
                           before_trade=lambda store: calls.append(('prepare', now[0])))
    monkeypatch.setattr('trading.scheduler.is_current_trading', lambda now: False)
    sch.start_check_trading()
    assert not sch.data_ready and len(calls) == 1
    now[0] += timedelta(minutes=1)
    sch.start_check_trading()
    assert len(calls) == 1
    now[0] += timedelta(minutes=4)
    sch.start_check_trading()
    assert sch.data_ready and len(calls) == 2
    now[0] = datetime(2026, 9, 11, 9, 25, 10)
    sch.start_check_trading()
    assert [c[0] for c in calls] == ['check', 'check', 'prepare']
    now[0] = datetime(2026, 9, 12, 0, 0)
    sch.start_check_trading()
    assert [c[0] for c in calls] == ['check', 'check', 'prepare', 'check']


def test_scheduler_failed_preflight_blocks_new_plan(monkeypatch):
    from trading.scheduler import TradingScheduler
    sch = TradingScheduler(object(), time_provider=lambda: datetime(2026, 9, 11, 9, 25, 10),
                           fast_forward=True, fast_forward_through_day=False,
                           ensure_data=lambda store: False,
                           before_trade=lambda store: pytest.fail('数据失败不能选股'),
                           execute_trade=lambda store: pytest.fail('数据失败不能下单'))
    monkeypatch.setattr('trading.scheduler.is_current_trading', lambda now: False)
    sch.start_check_trading()
    assert not sch.prepared and not sch.executed


def test_startup_checkpoint_does_not_complete_evening_update(tmp_path, monkeypatch):
    from data import update_all
    from trading import post_close
    monkeypatch.setattr(update_all, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(update_all, 'current_update_date', lambda: date(2026, 9, 11))
    calls = []
    for name in ('_update_stock_list', '_update_stock_name', '_update_delist',
                 '_update_balance', '_update_financial_deep', '_update_issue_price',
                 '_update_trading_calendar'):
        monkeypatch.setattr(update_all, name, lambda **kwargs: None)
    monkeypatch.setattr(update_all, '_update_kline', lambda **kw: calls.append(('kline', kw)))
    monkeypatch.setattr(update_all, '_update_indices', lambda **kw: calls.append(('indices', kw)))
    path = tmp_path / 'runtime' / 'runtime_2026-06-01_2026-09-10.npz'

    def build(**kwargs):
        calls.append(('build', kwargs))
        return write_runtime(path)

    monkeypatch.setattr(update_all, '_build_runtime', build)
    assert post_close.run_update_all(store(), history_end=PREVIOUS, history_days=70)
    checkpoint = tmp_path / 'runtime' / '.update_checkpoints' / '2026-09-11'
    assert not (checkpoint / 'final.done').exists()
    assert (checkpoint / 'startup_2026-09-10' / 'final.done').exists()
    assert (checkpoint / 'startup_2026-09-10' / 'step_stock_status.done').exists()
    assert not (checkpoint / 'startup_2026-09-10' / 'step_stock_name.done').exists()
    assert calls == [('kline', {'anchor_date': PREVIOUS, 'days': 70}),
                     ('indices', {'end_date': PREVIOUS}), ('build', {'end_date': PREVIOUS})]
    # final 存在但文件损坏时必须重新构建，不能一直跳过。
    write_runtime(path, drop='open')
    assert post_close.run_update_all(store(), history_end=PREVIOUS, history_days=70)
    assert len([c for c in calls if c[0] == 'build']) == 2


@pytest.mark.parametrize('full', [False, True])
def test_history_download_excludes_current_session(tmp_path, monkeypatch, full):
    from data import kline_mootdx as km
    raw = pd.DataFrame({'time': [int(pd.Timestamp(d).timestamp() * 1000)
                                for d in ['2026-09-09', '2026-09-10', '2026-09-11']]})
    for field in ['open', 'high', 'low', 'close', 'volume', 'amount', 'preClose']:
        raw[field] = 1.0
    mdx = SimpleNamespace(bars=lambda **kw: raw, xdxr=lambda **kw: pd.DataFrame(), close=lambda: None)
    monkeypatch.setattr(km, 'RAW_DIR', tmp_path)
    monkeypatch.setattr(km, '_connect_mootdx', lambda: mdx)
    monkeypatch.setattr(km, '_mootdx_bars_to_df', lambda *a, **kw: raw.copy())
    monkeypatch.setattr(km, '_fetch_bars_all', lambda *a: raw)
    path = tmp_path / '000001.SZ.parquet'
    if full:
        km.update_full(start='20260909', end='20260910', codes=['000001.SZ'])
    else:
        raw.to_parquet(path)
        result = km.update_recent(3, anchor_date=PREVIOUS, codes=['000001.SZ'], collect=True)
        assert max(result['000001.SZ']['time']) == raw['time'][1]
    assert pd.read_parquet(path)['time'].tolist() == raw['time'][:2].tolist()


def test_startup_indices_are_refreshed_again_after_close(tmp_path, monkeypatch):
    import akshare as ak
    from data import update_all
    monkeypatch.setattr(update_all, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(update_all, 'current_update_date', lambda: date(2026, 9, 11))
    calls = []

    def fetch(symbol):
        calls.append(symbol)
        return pd.DataFrame({'date': ['2026-09-09', '2026-09-10', '2026-09-11'],
                             'open': [1., 2., 3.]})

    monkeypatch.setattr(ak, 'stock_zh_index_daily', fetch)
    update_all._update_indices(end_date=PREVIOUS)
    for path in tmp_path.glob('index_*.parquet'):
        assert pd.to_datetime(pd.read_parquet(path)['trade_date']).max().date() == PREVIOUS
    update_all._update_indices()
    assert len(calls) == 6
    for path in tmp_path.glob('index_*.parquet'):
        assert pd.to_datetime(pd.read_parquet(path)['trade_date']).max().date() == date(2026, 9, 11)


def test_existing_execution_is_restored_before_slow_repair(monkeypatch):
    from trading.scheduler import TradingScheduler
    calls = []
    sch = TradingScheduler(object(), time_provider=lambda: datetime(2026, 9, 11, 9, 30),
                           fast_forward=True, fast_forward_through_day=False,
                           ensure_data=lambda store: calls.append('repair') or True,
                           execute_trade=lambda store: calls.append('execute'))
    sch.prepared = True
    sch._resume_execution = True
    monkeypatch.setattr('trading.scheduler.is_current_trading', lambda now: False)
    sch.start_check_trading()
    assert calls == ['execute']
    sch.start_check_trading()
    assert calls == ['execute', 'repair']


@pytest.mark.parametrize('facts_ok', [False, True])
def test_late_start_captures_facts_before_repair(monkeypatch, facts_ok):
    from trading.scheduler import TradingScheduler
    calls = []

    def post_close(store):
        calls.append('facts')
        store._close_facts_ready = facts_ok
        return False  # 分析失败也不应阻塞已采集事实后的补数。

    sch = TradingScheduler(object(), time_provider=lambda: datetime(2026, 9, 11, 15, 40),
                           fast_forward=True, fast_forward_through_day=False,
                           ensure_data=lambda store: calls.append('repair') or True,
                           post_close=post_close)
    monkeypatch.setattr('trading.scheduler.is_current_trading', lambda now: False)
    sch.start_check_trading()
    assert calls == (['facts', 'repair'] if facts_ok else ['facts'])


def test_calendar_is_refreshed_before_resolving_previous_session(tmp_path, runtime_path, monkeypatch):
    from data import update_all
    from utils.stock import time as stock_time
    import akshare as ak
    monkeypatch.setattr(update_all, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(update_all, 'current_update_date', lambda: date(2026, 9, 11))
    stale = (frozenset([date(2026, 9, 8)]), date(2026, 9, 8), date(2026, 9, 8))
    monkeypatch.setattr(stock_time, '_TRADING_CALENDAR_STATE', stale)
    pd.DataFrame({'trade_date': [date(2026, 9, 8)]}).to_parquet(tmp_path / 'trading_calendar.parquet')
    calls = []

    def fetch():
        calls.append('calendar')
        return pd.DataFrame({'trade_date': [PREVIOUS, date(2026, 12, 31)]})

    monkeypatch.setattr(ak, 'tool_trade_date_hist_sina', fetch)
    monkeypatch.setattr(startup, 'validate_startup_runtime', lambda previous: calls.append(previous))
    assert startup.ensure_startup_data(store())
    assert calls == ['calendar', PREVIOUS]
    assert stock_time._TRADING_CALENDAR_STATE != stale


def test_weekend_evening_start_checks_data_immediately(monkeypatch):
    from trading.scheduler import TradingScheduler
    calls = []
    sch = TradingScheduler(object(), time_provider=lambda: datetime(2026, 9, 12, 17, 0),
                           fast_forward=True, fast_forward_through_day=False,
                           ensure_data=lambda store: calls.append('repair') or True,
                           post_close=lambda store: pytest.fail('周末不采集交易日盘后事实'))
    sch.start_check_trading()
    assert calls == ['repair'] and sch.data_ready
