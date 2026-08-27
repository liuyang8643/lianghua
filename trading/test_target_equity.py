"""目标权益锚定 T-1 基线 的单测 —— 保证「今日目标」与当天运行次数无关（幂等）。

回归 603810 case：重跑用实时持仓/资金重算 base_target → 追价 churn。
修复后 total_eq 只由 T-1 收盘基线决定，重跑收敛到同一目标。
"""
from datetime import date

import numpy as np
import pandas as pd

from core.strategy import build_rebalance_day
import trading.persistence as persistence
import utils.stock.time as stime
import trading.main as m


class _AlwaysOkChecker:
    def check(self, candidates_idx, trade_idx, signal_date, is_buy):
        return np.ones(len(candidates_idx), dtype=bool), candidates_idx


def _strategy_fixture():
    data = {
        'open': np.array([[10.0, 5.0, 20.0]], dtype=np.float64),
        'close': np.array([[10.0, 5.0, 20.0]], dtype=np.float64),
        'preClose': np.array([[10.0, 5.0, 20.0]], dtype=np.float64),
        'trade_dates': np.array(['2026-06-02'], dtype='datetime64[D]'),
    }
    valid_stocks = ['A.SZ', 'B.SH', 'C.SH']
    stock_indices = {c: i for i, c in enumerate(valid_stocks)}
    valid_cols = np.array([0, 1, 2], dtype=np.intp)
    all_scores = {'F': np.array([[0.9, 0.8, 0.7]], dtype=np.float32)}
    return data, all_scores, valid_stocks, stock_indices, valid_cols


def test_target_equity_independent_of_live_state():
    """给定 T-1 基线，total_eq 与「实时回退权益」无关 —— 重跑结果恒定。"""
    prev_cash = 50_000.0
    y_shares = {'A.SZ': 1000, 'B.SH': 2000}
    prices = {'A.SZ': 10.0, 'B.SH': 5.0}
    expected = 50_000.0 + 1000 * 10.0 + 2000 * 5.0  # 70,000
    # 模拟不同运行时刻的实时权益(早上 vs 重跑后),结果必须一致
    data, all_scores, valid_stocks, stock_indices, valid_cols = _strategy_fixture()
    common = dict(
        data=data, all_scores=all_scores, date_idx=0, trade_idx=0,
        signal_date=date(2026, 6, 2), valid_stocks=valid_stocks,
        valid_cols=valid_cols, stock_indices=stock_indices, weights={'F': 1.0},
        buy_n=2, sell_m=2, checker=_AlwaysOkChecker(),
        sellable_volumes={'A.SZ': 1000}, rebalance=True,
        target_cash=prev_cash, target_positions=y_shares,
    )
    p1 = build_rebalance_day(**common, positions={'A.SZ': 1000}, cash=123_456.0)
    p2 = build_rebalance_day(**common, positions={'A.SZ': 1000}, cash=987_654.0)
    assert prices == {'A.SZ': p1.prices['A.SZ'], 'B.SH': p1.prices['B.SH']}
    assert p1.total_eq == p2.total_eq == expected


def test_target_equity_falls_back_without_baseline():
    data, all_scores, valid_stocks, stock_indices, valid_cols = _strategy_fixture()
    p = build_rebalance_day(
        data=data, all_scores=all_scores, date_idx=0, trade_idx=0,
        signal_date=date(2026, 6, 2), valid_stocks=valid_stocks,
        valid_cols=valid_cols, stock_indices=stock_indices, weights={'F': 1.0},
        buy_n=2, sell_m=2, checker=_AlwaysOkChecker(),
        positions={}, sellable_volumes={}, cash=88_888.0, rebalance=True,
    )
    assert p.total_eq == 88_888.0


def test_missing_open_is_suspended_excluded_and_uses_preclose_for_equity():
    """T 日 open 缺失：原持仓不卖、候选不可买，估值使用官方前收。"""
    data, all_scores, valid_stocks, stock_indices, valid_cols = _strategy_fixture()
    data['open'] = np.array([
        [10.0, 5.0, 20.0],
        [np.nan, 5.5, 21.0],
    ], dtype=np.float64)
    data['close'] = np.array([
        [10.0, 5.0, 20.0],
        [np.nan, 5.5, 21.0],
    ], dtype=np.float64)
    data['preClose'] = np.array([
        [9.8, 4.9, 19.5],
        [10.0, 5.0, 20.0],
    ], dtype=np.float64)
    data['trade_dates'] = np.array(['2026-06-01', '2026-06-02'], dtype='datetime64[D]')
    all_scores = {'F': np.array([
        [0.9, 0.8, 0.7],
        [0.9, 0.8, 0.7],
    ], dtype=np.float32)}

    p = build_rebalance_day(
        data=data, all_scores=all_scores, date_idx=1, trade_idx=1,
        signal_date=date(2026, 6, 2), valid_stocks=valid_stocks,
        valid_cols=valid_cols, stock_indices=stock_indices, weights={'F': 1.0},
        buy_n=1, sell_m=1, checker=_AlwaysOkChecker(),
        positions={'A.SZ': 1000}, sellable_volumes={'A.SZ': 1000},
        cash=50_000.0, rebalance=True,
    )
    assert p.t1_ranking[0] == 'A.SZ'
    assert 'A.SZ' not in p.buy_n_stocks
    assert p.buy_n_stocks == ['B.SH']
    assert 'A.SZ' not in p.prices
    assert p.sell_orders == []
    assert set(p.buy_orders) == {'B.SH'}
    assert p.total_eq == 60_000.0


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


def test_load_prev_baseline_keeps_empty_position_cash(tmp_path, monkeypatch):
    prev = date(2026, 5, 30)
    monkeypatch.setattr(persistence, '_TRADE_DIR', tmp_path)
    monkeypatch.setattr(stime, 'get_last_trading_day', lambda d: prev)
    pd.DataFrame([
        {'date': prev, 'code': 'A.SZ', 'volume': 0},
    ]).to_parquet(tmp_path / f'positions_{prev.isoformat()}.parquet', index=False)
    pd.DataFrame([{'date': prev, 'cash': 50_000.0}]).to_parquet(
        tmp_path / 'daily_summary.parquet', index=False)

    cash, y = m._load_prev_eod_baseline(date(2026, 6, 2))
    assert cash == 50_000.0
    assert y == {}
