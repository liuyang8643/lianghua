from datetime import date, datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from trading.bridge_wire import atomic_json
from trading.file_post_close import mirror_previous_baseline


def test_today_facts_are_saved_before_missing_previous_baseline(tmp_path, monkeypatch):
    from trading import file_post_close, post_close
    events = []
    atomic_json(tmp_path / 'config.json', {'account': 'same'})
    snapshot = dict(captured_at=datetime(2026, 11, 3, 15, 35).timestamp())
    (tmp_path / 'journal').mkdir()
    atomic_json(tmp_path / 'journal/execution_2026-11-03.json', dict(account='same', id='id', pending={'trade_date':'2026-11-03'}))
    from trading import persistence
    monkeypatch.setattr(persistence, '_TRADE_DIR', tmp_path)
    monkeypatch.setattr(file_post_close, 'mirror_frozen_plan', lambda *a: None)
    import core.strategy_config
    monkeypatch.setattr(core.strategy_config, 'load_strategy_config', lambda *a: dict(individual_config={},factor_classes=[]))
    monkeypatch.setattr(post_close, 'run_post_close', lambda store: store._prepare_close_valuation())
    monkeypatch.setattr(file_post_close, 'FileAccount', lambda *a, **k:
                        SimpleNamespace(_snapshot=lambda: snapshot))
    monkeypatch.setattr(post_close, '_capture_live_close_facts', lambda *a, **k:
                        (events.append('today_facts') or {'ok': True}))
    def missing(*args, **kwargs):
        events.append('baseline'); raise FileNotFoundError('missing baseline')
    monkeypatch.setattr(file_post_close, 'mirror_previous_baseline', missing)
    with pytest.raises(FileNotFoundError):
        file_post_close.reconcile_from_files(date(2026, 11, 3), 'unused', tmp_path)
    assert events == ['today_facts', 'baseline']


def test_bound_baseline_preserves_unknown_history_and_asset_gap(tmp_path):
    directory = tmp_path / 'files'; directory.mkdir()
    audit = tmp_path / 'audit'; audit.mkdir()
    kline = tmp_path / 'k-line'; kline.mkdir()
    pd.DataFrame([dict(time=pd.Timestamp('2026-11-02 15:00'),close=11)]).to_parquet(kline / '600001.SH.parquet')
    facts = dict(version=1, account='same', trade_date='2026-11-02',
        captured_at=datetime(2026, 11, 2, 15, 35).timestamp(),
        asset=dict(cash=100, total_asset=1200, market_value=1000),
        positions=[dict(stock_code='600001.SH', volume=100, can_use_volume=100,
                        market_value=1000, avg_price=9)])
    atomic_json(directory / 'account_eod_2026-11-02.json', facts)
    for _ in range(2):
        mirror_previous_baseline(directory, 'same', date(2026, 11, 3), audit)
    positions = pd.read_parquet(audit / 'positions_2026-11-02.parquet')
    assert positions.daily_pnl.isna().all() and positions.bought_today.isna().all()
    assert positions.last_price.tolist() == [11]
    summary = pd.read_parquet(audit / 'daily_summary.parquet')
    assert len(summary) == 1 and summary.total_asset.iloc[0] == 1200
    proof = json.loads((audit / 'baseline_2026-11-02.json').read_text())
    assert proof['asset_component_gap'] == 0
    assert proof['valuation']['raw_asset_component_gap'] == 100
    with pytest.raises(ValueError, match='account/date'):
        mirror_previous_baseline(directory, 'other', date(2026, 11, 3), audit)
    summary.loc[0, 'cash'] = 101; summary.to_parquet(audit / 'daily_summary.parquet')
    with pytest.raises(ValueError, match='conflicts'):
        mirror_previous_baseline(directory, 'same', date(2026, 11, 3), audit)


def test_file_card_is_original_renderer_with_no_timer_or_network(tmp_path, monkeypatch):
    import trading.day_board as module
    from trading.daily_records import write_planner_record, write_executor_record
    from trading.test_daily_records import facts
    monkeypatch.setattr(module, 'get_stock_name_at_date', lambda code, day: code)
    monkeypatch.setattr(module.threading, 'Timer', lambda *a, **k: pytest.fail('Timer created'))
    monkeypatch.setattr(module.lark_sender, 'send_card', lambda *a, **k: pytest.fail('Network send'))
    class FixedClock(datetime):
        @classmethod
        def now(cls): return cls(2026, 11, 3, 15, 40)
    monkeypatch.setattr(module, 'datetime', FixedClock)
    doc, initial, journal, snapshot = facts()
    planner = json.loads(write_planner_record(tmp_path, doc, snapshot=initial).read_text())
    executor = json.loads(write_executor_record(tmp_path, journal, snapshot).read_text())
    close = dict(summary=dict(live_daily_pnl=-100, live_daily_return_pct=-1,
                             bt_daily_return_pct=-0.9), holdings=executor['actual_holdings'])
    card = module.build_file_daily_card(planner, executor, close)
    board = module.TradingDayBoard(auto_publish=False)
    board.start_session(trade_date=date(2026, 11, 3), plan_rows=module.file_plan_rows(planner),
        y_positions=planner['initial_holdings'], bt_ref=dict(
            buy={'600002.SH': dict(shares=100, amount=0)},
            sell={'600001.SH': dict(shares=200, amount=0)},
            positions={'600001.SH': 0, '600002.SH': 100, '159001.SZ': 100}))
    for row in snapshot['orders']:
        board.record_order(SimpleNamespace(price=0, traded_price=0, status_msg='', **row))
    for row in snapshot['trades']: board.record_trade(SimpleNamespace(**row))
    board._live_positions = executor['actual_holdings']
    board.feed_close_data(live_pnl=-100, live_return=-1)
    board.feed_bt_reference(board._bt_ref, bt_daily_return=-0.9, comparison_basis='同计划零磨损')
    board.finalize()
    assert card == board._build_card()
    assert card['header']['template'] == 'red'
    assert '已锁定' in card['body']['elements'][-1]['text']['content']
