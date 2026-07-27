from datetime import datetime

import numpy as np

from core.backtest import _backtest_direct


def _run_buffered_position(monkeypatch, multiplier, *, enforce=False):
    new_top = "000001.SZ"
    buffered = "000002.SZ"
    codes = [new_top, buffered]
    data = {
        "stock_codes": np.array(codes),
        "trade_dates": np.array(["2024-01-02"], dtype="datetime64[D]"),
        "open": np.array([[10.0, 10.0]]),
        "high": np.array([[10.1, 10.1]]),
        "low": np.array([[9.9, 9.9]]),
        "close": np.array([[10.0, 10.0]]),
        "volume": np.array([[1_000_000.0, 1_000_000.0]]),
        "preClose": np.array([[10.0, 10.0]]),
        "st_mask": np.array([[False, False]]),
        "issue_price": np.array([10.0, 10.0]),
    }
    monkeypatch.setattr("core.backtest.get_delist_stock_info", lambda: {})

    return _backtest_direct(
        data=data,
        all_scores={"Trend": np.array([[2.0, 1.0]], dtype=np.float32)},
        valid_dates=[datetime(2024, 1, 2)],
        date_indices=[0],
        valid_stocks=codes,
        stock_indices={code: i for i, code in enumerate(codes)},
        weights={"Trend": 1.0},
        buy_n=1,
        sell_m=2,
        position_multipliers=np.array([multiplier]),
        init_cash=1.0,
        init_total_asset=10_001.0,
        init_positions={buffered: {"volume": 1_000, "avg_price": 10.0}},
        lightweight=True,
        market_order_freeze=False,
        list_dates_map={},
        enforce_position_multiplier_on_sell_m=enforce,
    )


def test_sell_m_buffer_keeps_legacy_exemption_by_default(monkeypatch):
    result = _run_buffered_position(monkeypatch, 0.0)

    assert result["cleared_positions_count"] == 0
    assert result["daily_exposures"][0] > 0.99


def test_enforced_zero_multiplier_clears_sell_m_buffer(monkeypatch):
    result = _run_buffered_position(monkeypatch, 0.0, enforce=True)

    assert result["cleared_positions_count"] == 1
    assert result["daily_exposures"] == [0.0]


def test_enforcement_keeps_sell_m_buffer_when_multiplier_is_one(monkeypatch):
    result = _run_buffered_position(monkeypatch, 1.0, enforce=True)

    assert result["cleared_positions_count"] == 0
    assert result["daily_exposures"][0] > 0.99


def test_enforced_fractional_multiplier_replaces_buffer_with_scaled_topn(monkeypatch):
    result = _run_buffered_position(monkeypatch, 0.5, enforce=True)

    assert result["cleared_positions_count"] == 1
    assert 0.49 < result["daily_exposures"][0] < 0.51
