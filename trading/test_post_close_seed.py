from datetime import date
from types import SimpleNamespace

import pandas as pd

import pytest


def test_post_close_persists_broker_facts_before_backtest_failure(tmp_path, monkeypatch):
    from trading import persistence, post_close

    trade_date = date(2026, 9, 3)
    events = []

    class FakeTradeRecorder:
        def sync_bank_transfers_from_qmt(self, trader, trade_date):
            events.append('cash_flows')

        def backfill_fills_from_qmt(self, trader, trade_date):
            events.append('fills')
            return 0

        def get_today_fills_df(self, trade_date):
            return pd.DataFrame()

        def snapshot_positions(self, positions, fills_df, trade_date):
            events.append('positions')

        def write_daily_summary(self, **kwargs):
            events.append('daily_summary')

    class FakeTrader:
        def query_asset(self):
            events.append('asset')
            return SimpleNamespace(total_asset=1_000_000, cash=100_000, market_value=900_000)

        def query_positions(self):
            return [SimpleNamespace(stock_code='000001.SZ', volume=100)]

    class FakeRecorder:
        def mark(self, name):
            events.append(name)

    store = SimpleNamespace(
        trader=FakeTrader(),
        _now=lambda: SimpleNamespace(date=lambda: trade_date),
        _skip_update=False,
        _historical_replay=False,
        _individual_config={},
        _factor_classes=[],
    )
    monkeypatch.setattr(persistence, 'live_trade_recorder', FakeTradeRecorder())
    monkeypatch.setattr(persistence, '_TRADE_DIR', tmp_path)
    monkeypatch.setattr(post_close, 'recorder', FakeRecorder())
    monkeypatch.setattr(post_close, '_post_close_update_kline', lambda d: {})

    def failed_seed_replay(*args, **kwargs):
        events.append('backtest')
        raise RuntimeError('frozen replay unavailable')

    store._frozen_reference=failed_seed_replay
    with pytest.raises(RuntimeError,match='frozen replay unavailable'):
        post_close.run_post_close(store)
    assert events.index('asset') < events.index('backtest')
    assert events.index('fills') < events.index('backtest')
    assert events.index('positions') < events.index('backtest')
    assert events.index('daily_summary') < events.index('backtest')


def test_incomplete_broker_facts_stop_before_slow_analytics(tmp_path, monkeypatch):
    from trading import persistence, post_close

    trade_date = date(2026, 9, 3)
    slow_calls = []

    class FakeTradeRecorder:
        def sync_bank_transfers_from_qmt(self, trader, trade_date):
            return 0

        def backfill_fills_from_qmt(self, trader, trade_date):
            return None

        def get_today_fills_df(self, trade_date):
            return pd.DataFrame()

        def snapshot_positions(self, positions, fills_df, trade_date):
            raise AssertionError("unknown empty account must not overwrite positions")

    class MissingAccountTrader:
        def query_asset(self):
            return None

        def query_positions(self):
            return None

    store = SimpleNamespace(
        trader=MissingAccountTrader(),
        _now=lambda: SimpleNamespace(date=lambda: trade_date),
        _skip_update=False,
        _historical_replay=False,
        _individual_config={},
        _factor_classes=[],
    )
    monkeypatch.setattr(persistence, 'live_trade_recorder', FakeTradeRecorder())
    monkeypatch.setattr(persistence, '_TRADE_DIR', tmp_path)
    monkeypatch.setattr(post_close, '_post_close_update_kline', lambda d: slow_calls.append('kline'))

    assert post_close.run_post_close(store) is None
    assert slow_calls == []


def test_all_cash_account_persists_authoritative_empty_positions(tmp_path, monkeypatch):
    from trading import persistence, post_close

    trade_date = date(2026, 9, 3)

    class FakeTradeRecorder:
        def sync_bank_transfers_from_qmt(self, trader, trade_date):
            return 0

        def backfill_fills_from_qmt(self, trader, trade_date):
            return 0

        def get_today_fills_df(self, trade_date):
            return pd.DataFrame()

        def snapshot_positions(self, positions, fills_df, trade_date):
            return None

        def write_daily_summary(self, **kwargs):
            return None

    class AllCashTrader:
        def query_asset(self):
            return SimpleNamespace(total_asset=100_000, cash=100_000, market_value=0)

        def query_positions(self):
            return []

    monkeypatch.setattr(persistence, 'live_trade_recorder', FakeTradeRecorder())
    monkeypatch.setattr(persistence, '_TRADE_DIR', tmp_path)

    facts = post_close._capture_live_close_facts(
        SimpleNamespace(trader=AllCashTrader()), trade_date, skip_account=False)

    positions_path = tmp_path / f'positions_{trade_date.isoformat()}.parquet'
    assert facts['ok'] is True
    assert positions_path.exists()
    assert pd.read_parquet(positions_path).empty
