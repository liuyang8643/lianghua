from datetime import datetime

import numpy as np

from core.backtest import (
    _backtest_direct,
    _compute_rebalance_funds_ratio,
    run_live_simulation,
)


def test_rebalance_funds_ratio_counts_capital_once_and_caps_at_one():
    sells = [{'shares': 80, 'price': 10.0}]
    buys = [{'shares': 70, 'price': 10.0}]

    assert _compute_rebalance_funds_ratio(sells, buys, 1_000.0) == 0.8
    assert _compute_rebalance_funds_ratio([], [], 1_000.0) == 0.0
    assert _compute_rebalance_funds_ratio(sells, buys, 0.0) == 0.0
    assert _compute_rebalance_funds_ratio(sells, buys, 500.0) == 1.0


def test_lightweight_backtest_returns_actual_daily_exposure(monkeypatch):
    code = "000001.SZ"
    data = {
        "stock_codes": np.array([code]),
        "trade_dates": np.array(["2024-01-02", "2024-01-03"], dtype="datetime64[D]"),
        "open": np.array([[10.0], [10.0]]),
        "high": np.array([[10.1], [10.1]]),
        "low": np.array([[9.9], [9.9]]),
        "close": np.array([[10.0], [10.0]]),
        "volume": np.array([[1_000_000.0], [1_000_000.0]]),
        "preClose": np.array([[10.0], [10.0]]),
        "st_mask": np.array([[False], [False]]),
        "issue_price": np.array([10.0]),
    }
    monkeypatch.setattr("core.backtest.get_delist_stock_info", lambda: {})

    result = _backtest_direct(
        data=data,
        all_scores={"Trend": np.array([[1.0], [1.0]], dtype=np.float32)},
        valid_dates=[datetime(2024, 1, 2), datetime(2024, 1, 3)],
        date_indices=[0, 1],
        valid_stocks=[code],
        stock_indices={code: 0},
        weights={"Trend": 1.0},
        buy_n=1,
        sell_m=1,
        lightweight=True,
        market_order_freeze=False,
        filter_masks={"signal": np.array([[True], [True]])},
        list_dates_map={},
    )

    assert len(result["daily_exposures"]) == 2
    assert len(result["daily_assets"]) == 2
    assert all(0.90 < value <= 1.0 for value in result["daily_exposures"])
    assert all(value > 0.0 for value in result["daily_assets"])


def test_live_simulation_accepts_json_stock_pool_list(monkeypatch):
    code = "000001.SZ"
    data = {
        "stock_codes": np.array([code]),
        "trade_dates": np.array(["2025-12-10"], dtype="datetime64[D]"),
        "open": np.array([[10.0]]),
    }
    monkeypatch.setattr("core.backtest._build_minute_lookup", lambda: {})
    monkeypatch.setattr("core.backtest._compute_timing_multipliers", lambda *args: None)
    monkeypatch.setattr(
        "core.backtest._backtest_direct",
        lambda *args, **kwargs: {"daily_returns": np.array([0.0])},
    )

    results = run_live_simulation(
        data=data,
        all_scores={"Trend": np.array([[1.0]], dtype=np.float32)},
        filter_masks={},
        stock_codes=data["stock_codes"],
        all_valid_stocks=[code],
        individuals=[{
            "stock_pool": ["60", "00", "30", "688"],
            "weights": {"Trend": 1.0},
            "buy_n": 1,
            "sell_m": 1,
        }],
        list_dates_map={},
        logger=type("Logger", (), {"info": lambda *args: None})(),
    )

    assert len(results) == 1
    assert set(results[0]["prices"]) == {"base", "09:32", "09:33", "09:34", "09:35"}


def test_non_rebalance_day_skips_prefilter_ranking(monkeypatch):
    code = "000001.SZ"
    data = {
        "stock_codes": np.array([code]),
        "trade_dates": np.array(["2024-01-02"], dtype="datetime64[D]"),
        "open": np.array([[10.0]]),
        "high": np.array([[10.1]]),
        "low": np.array([[9.9]]),
        "close": np.array([[10.0]]),
        "volume": np.array([[1_000_000.0]]),
        "preClose": np.array([[10.0]]),
        "st_mask": np.array([[False]]),
        "issue_price": np.array([10.0]),
    }
    monkeypatch.setattr("core.backtest.get_delist_stock_info", lambda: {})
    monkeypatch.setattr(
        "core.backtest._build_t1_ranking",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ranked")),
    )

    result = _backtest_direct(
        data=data,
        all_scores={"Trend": np.array([[1.0]], dtype=np.float32)},
        valid_dates=[datetime(2024, 1, 2)],
        date_indices=[0],
        valid_stocks=[code],
        stock_indices={code: 0},
        weights={"Trend": 1.0},
        buy_n=1,
        sell_m=1,
        holding_period=20,
        rebalance_start_index=1,
        lightweight=True,
        prefilter_n=1,
        list_dates_map={},
    )

    assert result["daily_exposures"] == [0.0]
