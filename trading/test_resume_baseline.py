from copy import deepcopy
from datetime import date, datetime, timedelta
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from trading.resume_baseline import load_resume_budget, prepare_request


DAY = date(2026, 11, 10)
NOW = datetime(2026, 11, 10, 9, 25, 12)


@pytest.fixture
def recovery(tmp_path):
    directory = tmp_path / 'data/live_file'
    directory.mkdir(parents=True)
    strategy = tmp_path / 'configs/config.json'
    strategy.parent.mkdir()
    strategy.write_text('{}')
    (directory / 'config.json').write_text(json.dumps(dict(account='TEST', trading_dates=[str(DAY)])))
    snapshot = dict(version=1, account='TEST', trade_date=str(DAY), captured_at=NOW.timestamp(),
                    asset=dict(account_id='TEST', cash=1000., market_value=1000., total_asset=2000., frozen_cash=0.),
                    positions=[dict(account_id='TEST', stock_code='600001.SH', volume=100,
                                    can_use_volume=100, market_value=1000.)],
                    orders=[], trades=[], trades_available=True)
    prepare_request(tmp_path, DAY, now=NOW)
    return tmp_path, directory, strategy, snapshot


def test_recovery_freezes_today_facts_without_forging_eod(recovery):
    root, directory, strategy, snapshot = recovery
    cash, positions, metadata = load_resume_budget(directory, snapshot, strategy, NOW)
    assert cash == 1000 and positions == {'600001.SH': 100}
    assert metadata['trade_date'] == str(DAY)
    assert not metadata['missing_period_pnl_verified']
    assert not list(directory.glob('account_eod_*.json'))
    path = directory / metadata['path']
    before = path.read_bytes()
    fresh = deepcopy(snapshot)
    fresh['captured_at'] += 2
    assert load_resume_budget(directory, fresh, strategy, NOW + timedelta(seconds=2))[2] == metadata
    assert path.read_bytes() == before


@pytest.mark.parametrize('change', ['stale', 'future', 'wrong_account', 'wrong_position_account',
                                  'unavailable_trades', 'missing_orders', 'malformed_orders', 'same_day_order',
                                  'same_day_trade', 'unknown_order', 'frozen_cash', 'missing_frozen_cash',
                                  'nan_frozen_cash', 'negative_frozen_cash', 'simulation'])
def test_invalid_recovery_facts_block(recovery, change):
    _, directory, strategy, s = recovery
    if change == 'stale': s['captured_at'] -= 31
    elif change == 'future': s['captured_at'] += 1
    elif change == 'wrong_account': s['asset']['account_id'] = 'OTHER'
    elif change == 'wrong_position_account': s['positions'][0]['account_id'] = 'OTHER'
    elif change == 'unavailable_trades': s['trades_available'] = False
    elif change == 'missing_orders': s['orders'] = None
    elif change == 'malformed_orders': s['orders'] = {}
    elif change == 'same_day_order': s['orders'] = [dict(account_id='TEST', order_time=NOW.timestamp(), order_status=54)]
    elif change == 'same_day_trade': s['trades'] = [dict(account_id='TEST', traded_time=NOW.timestamp())]
    elif change == 'unknown_order': s['orders'] = [dict(account_id='TEST', order_time=(NOW-timedelta(days=1)).timestamp(), order_status=255)]
    elif change == 'frozen_cash': s['asset']['frozen_cash'] = 100
    elif change == 'missing_frozen_cash': del s['asset']['frozen_cash']
    elif change == 'nan_frozen_cash': s['asset']['frozen_cash'] = float('nan')
    elif change == 'negative_frozen_cash': s['asset']['frozen_cash'] = -1
    elif change == 'simulation': s['simulation'] = True
    with pytest.raises((ValueError, KeyError)):
        load_resume_budget(directory, s, strategy, NOW)
    assert not list(directory.glob('budget_baseline_*.json'))


def test_changed_cash_cannot_rebase_a_frozen_budget(recovery):
    _, directory, strategy, snapshot = recovery
    load_resume_budget(directory, snapshot, strategy, NOW)
    snapshot['asset']['cash'] += 100
    snapshot['asset']['total_asset'] += 100
    with pytest.raises(ValueError, match='Account changed'):
        load_resume_budget(directory, snapshot, strategy, NOW)


def test_frozen_budget_retry_still_requires_known_zero_frozen_cash(recovery):
    _, directory, strategy, snapshot = recovery
    load_resume_budget(directory, snapshot, strategy, NOW)
    path = directory / f'budget_baseline_{DAY}.json'
    before = path.read_bytes()
    snapshot['asset']['frozen_cash'] = float('nan')
    with pytest.raises(ValueError, match='finite zero'):
        load_resume_budget(directory, snapshot, strategy, NOW)
    assert path.read_bytes() == before


def test_config_and_opening_time_are_bound(recovery):
    _, directory, strategy, snapshot = recovery
    with pytest.raises(ValueError, match='pre-open'):
        load_resume_budget(directory, snapshot, strategy, NOW.replace(hour=9, minute=30))
    strategy.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='matching'):
        load_resume_budget(directory, snapshot, strategy, NOW)


@pytest.mark.parametrize('filename', ['targets_2026-11-10.json', 'journal/execution_2026-11-10.json'])
def test_request_cannot_reset_existing_execution(recovery, filename):
    root, directory, _, _ = recovery
    path = directory / filename
    path.parent.mkdir(exist_ok=True)
    path.write_text('existing')
    with pytest.raises(RuntimeError, match='frozen plan'):
        prepare_request(root, DAY, now=NOW)
    assert path.read_text() == 'existing'


def test_publisher_uses_resume_budget_in_same_planner(recovery, monkeypatch):
    import core.backtest as backtest
    import core.legality as legality
    import core.strategy as strategy_module
    import core.strategy_config as config_module
    import data.db.stock_list as stock_list
    import trading.startup_data as startup
    import utils.stock.time as calendar
    from trading.plan_publisher import publish_target
    root, directory, strategy, snapshot = recovery
    (directory / 'account_latest.json').write_text(json.dumps(snapshot))
    monkeypatch.setattr(startup, 'validate_startup_runtime', lambda _: None)
    monkeypatch.setattr(calendar, '_get_trading_calendar_state', lambda: ({DAY}, DAY, DAY))
    monkeypatch.setattr(calendar, 'get_last_trading_day', lambda _: DAY-timedelta(days=1))
    config = dict(weights={'F': 1}, prefilter_n=0, buy_n=20, sell_m=25,
                  rebalance=True, cash_reserve_ratio=.15)
    monkeypatch.setattr(config_module, 'load_strategy_config', lambda _: dict(individual_config=config, factor_classes=[]))
    monkeypatch.setattr(stock_list, 'allow_buy_stock_code_list', lambda **_: ['600001.SH'])
    monkeypatch.setattr(backtest, '_compute_factor_scores', lambda *a, **k: ({}, {}, {}, [], [0], ['600001.SH'], {'600001.SH': 0}))
    monkeypatch.setattr(legality, 'LegalityChecker', lambda *a, **k: None)
    seen = {}
    def plan(**kw):
        seen.update(kw)
        return NS(prices={'600001.SH': 10.}, limit_prices={}, sell_orders=[],
                  buy_orders={}, buy_n_stocks=['600001.SH'])
    monkeypatch.setattr(strategy_module, 'build_rebalance_day', plan)
    path = publish_target(strategy, directory, now=NOW, overlay={'local': True}, simulation=True)
    result = json.loads(path.read_text())
    assert seen['target_cash'] == 1000 and seen['target_positions'] == {'600001.SH': 100}
    assert seen['position_multiplier'] == .85
    assert result['budget_baseline']['source'] == 'explicit_same_day_account_resume'
    assert result['simulation'] is True
    assert result['targets']['600001.SH']['shares'] == 100
    assert not list(directory.glob('account_eod_*.json'))


def test_terminal_status_matches_locked_sdk():
    from xtquant import xtconstant
    from trading.target_protocol import TERMINAL_ORDER_STATUSES
    assert TERMINAL_ORDER_STATUSES == {xtconstant.ORDER_SUCCEEDED, xtconstant.ORDER_CANCELED,
                                       xtconstant.ORDER_JUNK, xtconstant.ORDER_PART_CANCEL}


def test_latest_runtime_uses_content_date_not_filename(tmp_path, monkeypatch):
    import numpy as np
    import core.runtime as runtime
    monkeypatch.setattr(runtime, '_RUNTIME_DIR', tmp_path)
    old = tmp_path / 'runtime_1990_20261106_reference_complete_names.npz'
    latest = tmp_path / 'runtime_1990-12-19_2026-11-09.npz'
    np.savez(old, trade_dates=np.array(['2026-11-06'], dtype='datetime64[D]'))
    np.savez(latest, trade_dates=np.array(['2026-11-09'], dtype='datetime64[D]'))
    assert runtime.latest_runtime_npz_path() == latest
