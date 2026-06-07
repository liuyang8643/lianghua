"""daily_summary 的 daily_pnl 口径回归：

总盈亏以「个股盈亏总和」为准（免疫未记账的银证出入金），账户口径另存 account_pnl；
个股口径缺失（per_stock_pnl=None）时回退账户口径 (total_asset - prev_asset - net_cf)。
"""
from datetime import date

import pandas as pd
import pytest

import trading.persistence as persistence
from trading.persistence import LiveTradeRecorder


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    monkeypatch.setattr(persistence, "_TRADE_DIR", tmp_path)
    return LiveTradeRecorder()


def _seed_prev_day(rec, total_asset=1_000_000.0):
    rec.write_daily_summary(total_asset=total_asset, cash=0.0,
                            market_value=total_asset, trade_date=date(2026, 6, 1))


def test_daily_pnl_prefers_per_stock_when_given(recorder, tmp_path):
    _seed_prev_day(recorder)
    # 当日账户增 50,800（其中含未记账的 5w 入金），个股口径只赚 800
    recorder.write_daily_summary(
        total_asset=1_050_800.0, cash=0.0, market_value=1_050_800.0,
        trade_date=date(2026, 6, 2), per_stock_pnl=800.0)

    df = pd.read_parquet(tmp_path / "daily_summary.parquet")
    row = df[df['date'] == date(2026, 6, 2)].iloc[0]
    assert abs(row['daily_pnl'] - 800.0) < 1e-6          # 台账 daily_pnl = 个股口径
    assert abs(row['account_pnl'] - 50_800.0) < 1e-6     # 账户口径另存
    assert abs(row['daily_return_pct'] - 800.0 / 1_000_000.0 * 100) < 1e-9


def test_daily_pnl_falls_back_to_account_when_none(recorder, tmp_path):
    _seed_prev_day(recorder)
    recorder.write_daily_summary(
        total_asset=1_000_500.0, cash=0.0, market_value=1_000_500.0,
        trade_date=date(2026, 6, 2), per_stock_pnl=None)

    df = pd.read_parquet(tmp_path / "daily_summary.parquet")
    row = df[df['date'] == date(2026, 6, 2)].iloc[0]
    # 无个股口径 → daily_pnl 回退账户口径，且与 account_pnl 一致
    assert abs(row['daily_pnl'] - 500.0) < 1e-6
    assert abs(row['account_pnl'] - 500.0) < 1e-6
