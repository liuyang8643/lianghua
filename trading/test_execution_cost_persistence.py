"""Focused live-fill execution-cost persistence and compatibility tests."""

import sys
from dataclasses import dataclass
from datetime import date
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

import trading.persistence as persistence
import trading.replay as replay
from core.fees import (
    COMMISSION_RATE,
    MIN_COMMISSION,
    STAMP_TAX_RATE,
    TRANSFER_FEE_RATE,
)
from trading.persistence import (
    EVT_TRADE,
    LiveTradeRecorder,
    normalize_fill_costs,
)
from trading.report import PostCloseReport


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    monkeypatch.setattr(persistence, "_TRADE_DIR", tmp_path)
    return LiveTradeRecorder()


def _record_trade(recorder, *, direction, price, amount, order_id, **fees):
    recorder.record_event(
        EVT_TRADE,
        trade_date=date(2026, 7, 27),
        code="000001.SZ",
        name="平安银行",
        direction=direction,
        traded_price=price,
        traded_volume=100,
        amount=amount,
        order_id=order_id,
        traded_id=f"T{order_id}",
        est_price=10.0,
        **fees,
    )


def test_estimated_fill_persists_fee_components_and_directional_slippage(
    recorder,
    tmp_path,
):
    _record_trade(
        recorder,
        direction="buy",
        price=10.1,
        amount=1010.0,
        order_id=1,
    )
    _record_trade(
        recorder,
        direction="sell",
        price=9.8,
        amount=980.0,
        order_id=2,
    )

    fills = pd.read_parquet(tmp_path / "fills_2026-07-27.parquet")
    buy = fills[fills["direction"] == "buy"].iloc[0]
    sell = fills[fills["direction"] == "sell"].iloc[0]

    buy_broker = round(max(1010.0 * COMMISSION_RATE, MIN_COMMISSION), 4)
    buy_transfer = round(1010.0 * TRANSFER_FEE_RATE, 4)
    assert buy["fee_source"] == "estimated"
    assert buy["broker_commission"] == pytest.approx(buy_broker)
    assert buy["transfer_fee"] == pytest.approx(buy_transfer)
    assert buy["stamp_tax"] == 0.0
    assert buy["fee_est"] == pytest.approx(
        round(buy_broker + buy_transfer, 4)
    )
    assert buy["slippage_cost"] == pytest.approx(10.0)
    assert buy["total_execution_cost"] == pytest.approx(
        buy["fee_est"] + 10.0
    )

    sell_broker = round(max(980.0 * COMMISSION_RATE, MIN_COMMISSION), 4)
    sell_transfer = round(980.0 * TRANSFER_FEE_RATE, 4)
    sell_stamp = round(980.0 * STAMP_TAX_RATE, 4)
    assert sell["fee_source"] == "estimated"
    assert sell["broker_commission"] == pytest.approx(sell_broker)
    assert sell["transfer_fee"] == pytest.approx(sell_transfer)
    assert sell["stamp_tax"] == pytest.approx(sell_stamp)
    assert sell["fee_est"] == pytest.approx(
        round(sell_broker + sell_transfer + sell_stamp, 4)
    )
    # Selling below the planned Open is an adverse positive cost.
    assert sell["slippage_cost"] == pytest.approx(20.0)
    assert sell["total_execution_cost"] == pytest.approx(
        sell["fee_est"] + 20.0
    )


def test_actual_components_are_preserved_and_mismatches_fail_loudly(
    recorder,
    tmp_path,
):
    _record_trade(
        recorder,
        direction="buy",
        price=10.2,
        amount=1020.0,
        order_id=3,
        broker_commission=0.7,
        transfer_fee=0.2,
        stamp_tax=0.0,
        fee_est=0.9,
    )
    row = pd.read_parquet(
        tmp_path / "fills_2026-07-27.parquet"
    ).iloc[0]
    assert row["fee_source"] == "actual"
    assert row["fee_est"] == pytest.approx(0.9)
    assert row["slippage_cost"] == pytest.approx(20.0)
    assert row["total_execution_cost"] == pytest.approx(20.9)

    with pytest.raises(ValueError, match="fee_est mismatch"):
        _record_trade(
            recorder,
            direction="buy",
            price=10.0,
            amount=1000.0,
            order_id=4,
            broker_commission=0.7,
            transfer_fee=0.2,
            stamp_tax=0.0,
            fee_est=1.0,
        )
    with pytest.raises(ValueError, match="total_execution_cost mismatch"):
        _record_trade(
            recorder,
            direction="buy",
            price=10.0,
            amount=1000.0,
            order_id=5,
            broker_commission=0.7,
            transfer_fee=0.2,
            stamp_tax=0.0,
            fee_est=0.9,
            total_execution_cost=99.0,
        )


def test_repeated_traded_id_is_not_counted_twice_in_disk_or_memory(
    recorder,
    tmp_path,
):
    kwargs = {
        "direction": "buy",
        "price": 10.1,
        "amount": 1010.0,
        "order_id": 31,
    }
    _record_trade(recorder, **kwargs)
    _record_trade(recorder, **kwargs)

    disk = pd.read_parquet(tmp_path / "fills_2026-07-27.parquet")
    current = recorder.get_today_fills_df(date(2026, 7, 27))
    assert len(disk) == 1
    assert len(current) == 1
    assert len(recorder._today_fills) == 1
    assert current["fee_est"].sum() == pytest.approx(
        disk["fee_est"].sum()
    )


def test_restart_append_returns_complete_disk_history_without_duplicates(
    recorder,
    tmp_path,
):
    _record_trade(
        recorder,
        direction="buy",
        price=10.0,
        amount=1000.0,
        order_id=41,
    )

    restarted = LiveTradeRecorder()
    _record_trade(
        restarted,
        direction="sell",
        price=10.2,
        amount=1020.0,
        order_id=42,
    )
    fills = restarted.get_today_fills_df(date(2026, 7, 27))

    assert set(fills["traded_id"]) == {"T41", "T42"}
    assert len(fills) == 2
    assert len(restarted._today_fills) == 2
    disk = pd.read_parquet(tmp_path / "fills_2026-07-27.parquet")
    assert set(disk["traded_id"]) == {"T41", "T42"}


def test_restart_restores_planned_open_for_slippage(
    recorder,
    tmp_path,
):
    target = date(2026, 7, 27)
    recorder.record_plan([{
        "date": target,
        "code": "000001.SZ",
        "name": "平安银行",
        "direction": "buy",
        "est_price": 10.0,
        "est_volume": 100,
        "est_amount": 1000.0,
        "factor_score": 1.0,
        "limit_status": "normal",
        "reason": "",
        "plan_seq": 1,
    }], trade_date=target)

    restarted = LiveTradeRecorder()
    restarted.record_event(
        EVT_TRADE,
        trade_date=target,
        code="000001.SZ",
        direction="buy",
        traded_price=10.2,
        traded_volume=100,
        amount=1020.0,
        order_id=51,
        traded_id="T51",
    )

    row = pd.read_parquet(
        tmp_path / "fills_2026-07-27.parquet"
    ).iloc[0]
    assert row["est_price"] == pytest.approx(10.0)
    assert row["slippage_cost"] == pytest.approx(20.0)


def test_qmt_backfill_after_restart_restores_planned_open(
    recorder,
    tmp_path,
    monkeypatch,
):
    target = date(2026, 7, 27)
    recorder.record_plan([{
        "date": target,
        "code": "000001.SZ",
        "direction": "sell",
        "est_price": 10.0,
    }], trade_date=target)

    fake_xtquant = ModuleType("xtquant")
    fake_xtquant.xtconstant = SimpleNamespace(
        STOCK_BUY=23,
        STOCK_SELL=24,
    )
    monkeypatch.setitem(sys.modules, "xtquant", fake_xtquant)
    trade = SimpleNamespace(
        order_id=61,
        traded_id="Q61",
        stock_code="000001.SZ",
        order_type=24,
        traded_volume=100,
        traded_price=9.8,
        traded_amount=980.0,
    )
    trader = SimpleNamespace(query_all_trades=lambda: [trade])

    restarted = LiveTradeRecorder()
    assert restarted.backfill_fills_from_qmt(
        trader,
        trade_date=target,
    ) == 1
    row = pd.read_parquet(
        tmp_path / "fills_2026-07-27.parquet"
    ).iloc[0]
    assert row["est_price"] == pytest.approx(10.0)
    assert row["slippage_cost"] == pytest.approx(20.0)


def test_legacy_fill_upgrade_preserves_fee_and_computes_exact_slippage():
    legacy = pd.DataFrame([{
        "date": date(2026, 7, 27),
        "code": "000001.SZ",
        "name": "平安银行",
        "direction": "sell",
        "price": 9.8,
        "shares": 100,
        "amount": 980.0,
        "fee_est": 1.5,
        "order_id": 1,
        "traded_id": "",
        "fill_time": None,
        "est_price": 10.0,
        "slippage_pct": -2.0,
    }])

    upgraded = normalize_fill_costs(legacy)
    row = upgraded.iloc[0]
    assert row["fee_source"] == "legacy"
    assert row["broker_commission"] == pytest.approx(1.5)
    assert row["transfer_fee"] == 0.0
    assert row["stamp_tax"] == 0.0
    assert row["fee_est"] == pytest.approx(1.5)
    assert row["slippage_cost"] == pytest.approx(20.0)
    assert row["total_execution_cost"] == pytest.approx(21.5)

    corrupted = upgraded.copy()
    corrupted.loc[0, "total_execution_cost"] = 22.0
    with pytest.raises(ValueError, match="total_execution_cost mismatch"):
        normalize_fill_costs(corrupted)


@dataclass
class _Position:
    stock_code: str
    volume: int
    avg_price: float
    last_price: float
    can_use_volume: int = 100

    @property
    def market_value(self):
        return self.volume * self.last_price


def test_slippage_is_diagnostic_and_not_deducted_twice(
    recorder,
    tmp_path,
):
    fills = normalize_fill_costs(pd.DataFrame([{
        "code": "000001.SZ",
        "name": "平安银行",
        "direction": "buy",
        "price": 11.0,
        "shares": 100,
        "amount": 1100.0,
        "fee_est": 1.0,
        "broker_commission": 1.0,
        "transfer_fee": 0.0,
        "stamp_tax": 0.0,
        "slippage_cost": 100.0,
        "total_execution_cost": 101.0,
        "fee_source": "actual",
        "est_price": 10.0,
    }]))
    recorder.snapshot_positions(
        [_Position("000001.SZ", 100, 11.0, 11.0)],
        fills_df=fills,
        trade_date=date(2026, 7, 27),
    )
    row = pd.read_parquet(
        tmp_path / "positions_2026-07-27.parquet"
    ).iloc[0]
    # Actual price 11 is already the cash/P&L basis. Only explicit fee 1 is
    # deducted; the diagnostic slippage 100 must not be charged again.
    assert row["daily_pnl"] == pytest.approx(-1.0)


def test_report_and_replay_use_validated_persisted_costs(
    tmp_path,
    monkeypatch,
):
    legacy = pd.DataFrame([{
        "code": "000001.SZ",
        "name": "平安银行",
        "direction": "buy",
        "price": 10.1,
        "shares": 100,
        "amount": 1010.0,
        "fee_est": 0.2,
        "est_price": 10.0,
        "slippage_pct": 1.0,
    }])
    report = PostCloseReport(date(2026, 7, 27))
    report.feed_fills_df(legacy)
    costs = report.build_dim4_slippage()
    assert costs["total_fee_est"] == pytest.approx(0.2)
    assert costs["total_slippage_cost"] == pytest.approx(10.0)
    assert costs["total_execution_cost"] == pytest.approx(10.2)
    assert costs["fee_sources"] == {"legacy": 1}

    monkeypatch.setattr(replay, "_TRADE_DIR", tmp_path)
    legacy.to_parquet(tmp_path / "fills_2026-07-27.parquet", index=False)
    replayed = replay._read_fills(date(2026, 7, 27))
    assert replayed.iloc[0]["fee_source"] == "legacy"
    card = replay._build_daily_card(
        date(2026, 7, 27),
        plan_df=None,
        fills_df=replayed,
        pos_df=None,
        summary=None,
    )
    assert "成交成本分项" in str(card)
    assert "legacy:1" in str(card)
