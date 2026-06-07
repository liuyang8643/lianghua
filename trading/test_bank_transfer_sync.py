"""sync_bank_transfers_from_qmt 单测：回看窗口 + 真实日期记账 + 去重 + 迟到补登。

聚焦回归：T 日 15:00 后入金当天查不到，次日盘后能否按真实日期补登（而非漏记或错记到次日）。
"""
from __future__ import annotations
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

import trading.persistence as P
from trading.persistence import LiveTradeRecorder


def _stream(*, success=True, balance=50000.0, direction='1',
            no='T001', d='20260603', t='153000', bank='ICBC', remark=''):
    return SimpleNamespace(
        success=success, balance=balance, transfer_direction=direction,
        transfer_no=no, date=d, time=t, bank_name=bank, remark=remark,
    )


class _FakeTrader:
    """按 [start,end] 窗口过滤返回流水，模拟 QMT 银证查询。"""
    def __init__(self, streams):
        self._streams = streams

    def query_bank_transfers(self, start_date: str, end_date: str) -> list:
        return [s for s in self._streams if start_date <= str(s.date) <= end_date]


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    """把持久化目录指向临时路径，隔离真实 data/live_trades。"""
    monkeypatch.setattr(P, '_TRADE_DIR', tmp_path)
    return LiveTradeRecorder()


def _read(tmp_path) -> pd.DataFrame:
    return pd.read_parquet(tmp_path / 'cash_flows.parquet')


def test_records_under_transfer_date_not_run_date(recorder, tmp_path):
    """流水按自身日期(06-03)记账，即便运行日是 06-04。"""
    trader = _FakeTrader([_stream(d='20260603', balance=50000.0)])
    n = recorder.sync_bank_transfers_from_qmt(trader, trade_date=date(2026, 6, 4), lookback_days=5)
    assert n == 1
    assert recorder.get_today_cash_flows(trade_date=date(2026, 6, 3)) == 50000.0
    assert recorder.get_today_cash_flows(trade_date=date(2026, 6, 4)) == 0.0


def test_dedup_by_transfer_no(recorder, tmp_path):
    """同一 transfer_no 重复同步只入库一次。"""
    trader = _FakeTrader([_stream(no='T001', d='20260603')])
    assert recorder.sync_bank_transfers_from_qmt(trader, trade_date=date(2026, 6, 4)) == 1
    assert recorder.sync_bank_transfers_from_qmt(trader, trade_date=date(2026, 6, 4)) == 0
    assert len(_read(tmp_path)) == 1


def test_dedup_fallback_when_no_transfer_no(recorder, tmp_path):
    """无 transfer_no 时按 (日期,金额) 兜底去重，避免回看窗口重复入库。"""
    trader = _FakeTrader([_stream(no='', d='20260603', balance=50000.0)])
    assert recorder.sync_bank_transfers_from_qmt(trader, trade_date=date(2026, 6, 4)) == 1
    assert recorder.sync_bank_transfers_from_qmt(trader, trade_date=date(2026, 6, 4)) == 0
    assert len(_read(tmp_path)) == 1


def test_late_deposit_backfilled_next_day(recorder, tmp_path):
    """T 日 15:00 后入金：T 日盘后查不到(0条)，T+1 盘后按真实日期补登。"""
    # T 日(06-03)盘后：该笔流水尚未出现（模拟 15:00 后入金）
    trader_t = _FakeTrader([])
    assert recorder.sync_bank_transfers_from_qmt(trader_t, trade_date=date(2026, 6, 3)) == 0
    assert recorder.get_today_cash_flows(trade_date=date(2026, 6, 3)) == 0.0

    # T+1(06-04)盘后：回看窗口覆盖到 06-03，流水已出现 → 补登到 06-03
    trader_t1 = _FakeTrader([_stream(no='T777', d='20260603', balance=50000.0)])
    assert recorder.sync_bank_transfers_from_qmt(trader_t1, trade_date=date(2026, 6, 4), lookback_days=5) == 1
    assert recorder.get_today_cash_flows(trade_date=date(2026, 6, 3)) == 50000.0


def test_outflow_is_negative(recorder, tmp_path):
    """出金(direction='2')记为负数。"""
    trader = _FakeTrader([_stream(no='T002', d='20260603', direction='2', balance=20000.0)])
    recorder.sync_bank_transfers_from_qmt(trader, trade_date=date(2026, 6, 4))
    assert recorder.get_today_cash_flows(trade_date=date(2026, 6, 3)) == -20000.0


def test_skips_unsuccessful_and_nonpositive(recorder, tmp_path):
    """跳过 success=False 与非正金额的占位流水。"""
    trader = _FakeTrader([
        _stream(no='A', d='20260603', success=False),
        _stream(no='B', d='20260603', balance=0.0),
    ])
    assert recorder.sync_bank_transfers_from_qmt(trader, trade_date=date(2026, 6, 4)) == 0


def test_no_streams_returns_zero(recorder, tmp_path):
    trader = _FakeTrader([])
    assert recorder.sync_bank_transfers_from_qmt(trader, trade_date=date(2026, 6, 4)) == 0


def test_record_cash_flow_historical_date(recorder, tmp_path):
    """record_cash_flow(trade_date=...) 可补登历史日。"""
    d = date(2026, 6, 3)
    recorder.record_cash_flow(50000.0, flow_type='deposit',
                              note='补登测试', trade_date=d)
    assert recorder.get_today_cash_flows(trade_date=d) == 50000.0
    assert recorder.get_today_cash_flows(trade_date=date(2026, 6, 4)) == 0.0
    df = pd.read_parquet(tmp_path / 'cash_flows.parquet')
    assert len(df) == 1
    assert df['date'].iloc[0] == d or str(df['date'].iloc[0]) == '2026-06-03'
