"""目标权益锚定 T-1 基线 的单测 —— 保证「今日目标」与当天运行次数无关（幂等）。

回归 603810 case：重跑用实时持仓/资金重算 base_target → 追价 churn。
修复后 total_eq 只由 T-1 收盘基线决定，重跑收敛到同一目标。
"""
from datetime import date

import pandas as pd

import trading.main as m
import trading.persistence as persistence
import utils.stock.time as stime


def test_target_equity_independent_of_live_state():
    """给定 T-1 基线，total_eq 与「实时回退权益」无关 —— 重跑结果恒定。"""
    prev_cash = 50_000.0
    y_shares = {'A.SZ': 1000, 'B.SH': 2000}
    prices = {'A.SZ': 10.0, 'B.SH': 5.0}
    expected = 50_000.0 + 1000 * 10.0 + 2000 * 5.0  # 70,000
    # 模拟不同运行时刻的实时权益(早上 vs 重跑后),结果必须一致
    e1 = m._target_equity(prev_cash, y_shares, prices, live_fallback_eq=123_456.0)
    e2 = m._target_equity(prev_cash, y_shares, prices, live_fallback_eq=987_654.0)
    assert e1 == e2 == expected


def test_target_equity_falls_back_without_baseline():
    assert m._target_equity(None, None, {}, live_fallback_eq=88_888.0) == 88_888.0
    # 空持仓字典也回退实时
    assert m._target_equity(50_000.0, {}, {}, live_fallback_eq=88_888.0) == 88_888.0


def test_load_prev_baseline_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(persistence, '_TRADE_DIR', tmp_path)
    monkeypatch.setattr(stime, 'get_last_trading_day', lambda d: date(2026, 5, 30))
    assert m._load_prev_eod_baseline(date(2026, 6, 2)) == (None, None)


def test_load_prev_baseline_reads_snapshot(tmp_path, monkeypatch):
    prev = date(2026, 5, 30)
    monkeypatch.setattr(persistence, '_TRADE_DIR', tmp_path)
    monkeypatch.setattr(stime, 'get_last_trading_day', lambda d: prev)
    pd.DataFrame([
        {'date': prev, 'code': 'A.SZ', 'volume': 1000},
        {'date': prev, 'code': 'B.SH', 'volume': 0},   # 已清空,应剔除
    ]).to_parquet(tmp_path / f'positions_{prev.isoformat()}.parquet', index=False)
    pd.DataFrame([{'date': prev, 'cash': 50_000.0}]).to_parquet(
        tmp_path / 'daily_summary.parquet', index=False)

    cash, y = m._load_prev_eod_baseline(date(2026, 6, 2))
    assert cash == 50_000.0
    assert y == {'A.SZ': 1000}
