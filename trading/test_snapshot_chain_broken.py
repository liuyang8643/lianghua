"""snapshot_positions 链路断裂保护回归：

T-1 持仓快照缺失但存在更早快照（链路断裂）时，卖出仓位没有成本基线，
若仍按公式计算会把「卖出金额」整笔误当利润（曾导致单日 +58w / +81% 的假收益）。
此时 per-stock daily_pnl 必须标记为 None（不可计算），交由账户层口径兜底。

对照：真正首个交易日（无任何更早快照）→ 全是当日新开仓 → 正常计算。
"""
from dataclasses import dataclass
from datetime import date

import pandas as pd
import pytest

import trading.persistence as persistence
from trading.persistence import LiveTradeRecorder


@dataclass
class _Pos:
    stock_code: str
    volume: int
    avg_price: float
    last_price: float

    @property
    def market_value(self) -> float:
        return self.last_price * self.volume

    @property
    def can_use_volume(self) -> int:
        return self.volume


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    monkeypatch.setattr(persistence, "_TRADE_DIR", tmp_path)
    return LiveTradeRecorder()


def _fills(rows):
    return pd.DataFrame(rows, columns=[
        'code', 'direction', 'price', 'shares', 'amount', 'fee_est', 'name'])


def test_chain_broken_sold_to_zero_marks_none(recorder, tmp_path):
    # 存在更早快照 06-01，但 T-1(06-04) 缺失 → 06-05 链路断裂
    pd.DataFrame([{
        'code': '000001.SZ', 'name': 'A', 'volume': 1000,
        'last_price': 10.0, 'avg_price': 10.0, 'market_value': 10000.0,
    }]).to_parquet(tmp_path / "positions_2026-06-01.parquet")

    # 06-05 卖出 000001 全部 1000 股（成交 11000），持仓清零 → 无成本基线
    fills = _fills([('000001.SZ', 'sell', 11.0, 1000, 11000.0, 6.6, 'A')])
    recorder.snapshot_positions([_Pos('000001.SZ', 0, 0.0, 11.0)],
                                fills_df=fills, trade_date=date(2026, 6, 5))

    df = pd.read_parquet(tmp_path / "positions_2026-06-05.parquet")
    row = df[df['code'] == '000001.SZ'].iloc[0]
    # 已清仓且链路断裂 → daily_pnl 不可计算（绝不能算成 +11000）
    assert pd.isna(row['daily_pnl']), f"链路断裂清仓应标 None, 实得 {row['daily_pnl']}"


def test_chain_broken_held_uses_cost_basis(recorder, tmp_path):
    # 链路断裂下，仍持有的票应按成本基线 (close-avg)×vol - fee 计算持仓盈亏
    pd.DataFrame([{
        'code': '000001.SZ', 'name': 'A', 'volume': 1000,
        'last_price': 10.0, 'avg_price': 10.0, 'market_value': 10000.0,
    }]).to_parquet(tmp_path / "positions_2026-06-01.parquet")

    # 06-05 买入 000003 共 1000 股 @ 20（成本 20000, fee 2），收盘 19.5
    fills = _fills([('000003.SZ', 'buy', 20.0, 1000, 20000.0, 2.0, 'C')])
    recorder.snapshot_positions([_Pos('000003.SZ', 1000, 20.0, 19.5)],
                                fills_df=fills, trade_date=date(2026, 6, 5))

    df = pd.read_parquet(tmp_path / "positions_2026-06-05.parquet")
    row = df[df['code'] == '000003.SZ'].iloc[0]
    # mv 19500 - 成本 20000 - fee 2 = -502
    assert pd.notna(row['daily_pnl']), "持有仓位链路断裂下应可计算"
    assert abs(row['daily_pnl'] - (19500.0 - 20000.0 - 2.0)) < 1e-6


def test_first_day_computes_pnl(recorder, tmp_path):
    # 无任何更早快照 = 真正首日 → 当日买入应正常计算 daily_pnl
    fills = _fills([('000002.SZ', 'buy', 20.0, 500, 10000.0, 1.0, 'B')])
    recorder.snapshot_positions([_Pos('000002.SZ', 500, 20.0, 21.0)],
                                fills_df=fills, trade_date=date(2026, 6, 5))

    df = pd.read_parquet(tmp_path / "positions_2026-06-05.parquet")
    row = df[df['code'] == '000002.SZ'].iloc[0]
    # mv 21*500=10500, 买入成本 10000, fee 1 → ≈ +499
    assert pd.notna(row['daily_pnl'])
    assert abs(row['daily_pnl'] - (10500.0 - 10000.0 - 1.0)) < 1e-6
