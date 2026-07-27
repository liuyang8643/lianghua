from datetime import datetime

import numpy as np

from core.backtest import _backtest_direct
from testback.run_ga import _apply_exposure_constraint


def _one_stock_backtest(monkeypatch, multiplier, *, lightweight=True):
    code = "000001.SZ"
    data = {
        "stock_codes": np.array([code]),
        "stock_names": np.array(["测试股票"]),
        "trade_dates": np.array(["2024-01-02"], dtype="datetime64[D]"),
        "open": np.array([[10.0]]), "high": np.array([[10.1]]),
        "low": np.array([[9.9]]), "close": np.array([[10.0]]),
        "volume": np.array([[1_000_000.0]]), "preClose": np.array([[10.0]]),
        "st_mask": np.array([[False]]), "issue_price": np.array([10.0]),
    }
    monkeypatch.setattr("core.backtest.get_delist_stock_info", lambda: {})
    return _backtest_direct(
        data=data, all_scores={"Score": np.array([[1.0]], dtype=np.float32)},
        valid_dates=[datetime(2024, 1, 2)], date_indices=[0],
        valid_stocks=[code], stock_indices={code: 0}, weights={"Score": 1.0},
        buy_n=1, sell_m=1, position_multipliers=np.array([multiplier]),
        lightweight=lightweight, market_order_freeze=False, list_dates_map={},
    )


def test_lightweight_backtest_returns_actual_exposure(monkeypatch):
    invested = _one_stock_backtest(monkeypatch, 1.0)
    empty = _one_stock_backtest(monkeypatch, 0.0)

    assert len(invested["daily_assets"]) == 1
    assert 0.99 < invested["daily_exposures"][0] <= 1.0
    assert empty["daily_exposures"] == [0.0]


def test_detailed_snapshot_uses_same_exposure_and_turnover_contract(monkeypatch):
    result = _one_stock_backtest(monkeypatch, 1.0, lightweight=False)
    snapshot = result["daily_snapshots"][0]

    assert 0.99 < snapshot["exposure"] <= 1.0
    assert 0.99 < snapshot["rebalance_funds_ratio"] <= 1.0


def test_exposure_constraint_keeps_feasible_training_fitness():
    fitness, passed = _apply_exposure_constraint(
        2.5, 0.45, {"min_average_exposure": 0.45},
    )

    assert fitness == 2.5
    assert passed is True


def test_exposure_constraint_rejects_low_exposure_before_calmar():
    high_calmar, high_passed = _apply_exposure_constraint(
        9.0, 0.30, {"min_average_exposure": 0.45},
    )
    lower_calmar, lower_passed = _apply_exposure_constraint(
        1.0, 0.40, {"min_average_exposure": 0.45},
    )

    assert high_passed is False
    assert lower_passed is False
    assert lower_calmar > high_calmar
