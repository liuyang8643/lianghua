from datetime import date
from types import SimpleNamespace

import pandas as pd

from trading.post_close import _extract_live_seed


def _summary(prev_date, cash):
    return pd.DataFrame([{'date': prev_date, 'cash': cash, 'total_asset': cash}])


def test_extract_live_seed_basic():
    prev = date(2026, 5, 29)
    pos_df = pd.DataFrame([
        {'code': '600000.SH', 'volume': 1000, 'avg_price': 10.0, 'last_price': 11.0},
        {'code': '000001.SZ', 'volume': 500, 'avg_price': 20.0, 'last_price': 19.0},
        {'code': '300001.SZ', 'volume': 0, 'avg_price': 5.0, 'last_price': 5.0},  # 空仓行应被忽略
    ])
    seed = _extract_live_seed(pos_df, _summary(prev, 123_456.0), prev)
    assert seed is not None
    seed_cash, seed_positions, y_eod = seed

    assert seed_cash == 123_456.0
    assert set(seed_positions) == {'600000.SH', '000001.SZ'}
    assert seed_positions['600000.SH'] == {'volume': 1000, 'avg_price': 10.0}

    # T-1 收盘市值用 last_price 估
    mv = {p['code']: p['current_value'] for p in y_eod}
    assert mv['600000.SH'] == 1000 * 11.0
    assert mv['000001.SZ'] == 500 * 19.0


def test_extract_live_seed_returns_none_when_summary_missing_prev_date():
    prev = date(2026, 5, 29)
    pos_df = pd.DataFrame([
        {'code': '600000.SH', 'volume': 1000, 'avg_price': 10.0, 'last_price': 11.0},
    ])
    # daily_summary 没有 prev_date 行 → 无法确定种子现金
    other = _summary(date(2026, 5, 28), 100.0)
    assert _extract_live_seed(pos_df, other, prev) is None


def test_extract_live_seed_returns_none_when_no_positions():
    prev = date(2026, 5, 29)
    empty_pos = pd.DataFrame(columns=['code', 'volume', 'avg_price', 'last_price'])
    assert _extract_live_seed(empty_pos, _summary(prev, 100.0), prev) is None


def test_extract_live_seed_returns_none_when_all_positions_zero():
    prev = date(2026, 5, 29)
    pos_df = pd.DataFrame([
        {'code': '600000.SH', 'volume': 0, 'avg_price': 10.0, 'last_price': 11.0},
    ])
    assert _extract_live_seed(pos_df, _summary(prev, 100.0), prev) is None


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
        return None

    monkeypatch.setattr(post_close, '_run_seed_replay', failed_seed_replay)
    monkeypatch.setattr(post_close, '_load_latest_seed', lambda d: None)
    monkeypatch.setattr(post_close, '_resolve_backtest_start', lambda d: d)
    monkeypatch.setattr(post_close, '_run_continuous_backtest', lambda *a, **kw: None)

    assert post_close.run_post_close(store) is None
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
