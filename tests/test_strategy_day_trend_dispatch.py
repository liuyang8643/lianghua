from datetime import date, datetime

import numpy as np
import pytest

import core.backtest
import core.legality
import core.runtime
import core.strategy
import core.trend_timing


def _scored_history(trade_date):
    dates = np.datetime64("2024-01-01") + np.arange(25).astype("timedelta64[D]")
    data = {
        "trade_dates": dates,
        "stock_codes": np.array(["000001"]),
        "open": np.ones((25, 1), dtype=np.float64),
    }
    return data, (
        data,
        {"Factor": np.zeros((25, 1), dtype=np.float64)},
        {},
        [trade_date],
        [24],
        ["000001"],
        {"000001": 0},
    )


class _CompletedFactor:
    hist_days = 5

    def calc_batch(self, data):
        rows = len(data["trade_dates"])
        raw = np.tile(np.array([1.0, 2.0]), (rows, 1))
        raw[:self.hist_days] = np.nan
        return raw


class _AllowAllChecker:
    def __init__(self, *args, **kwargs):
        pass

    def check(self, stock_indices, *args, **kwargs):
        return np.ones(len(stock_indices), dtype=bool), {}


def test_build_strategy_day_dispatches_dual_completed(monkeypatch):
    trade_date = date(2024, 1, 25)
    data, scored = _scored_history(trade_date)
    monkeypatch.setattr(core.runtime, "load_runtime_npz", lambda *args, **kwargs: data)
    monkeypatch.setattr(core.backtest, "_compute_factor_scores", lambda *args, **kwargs: scored)
    monkeypatch.setattr(core.backtest, "_compute_list_dates", lambda *args, **kwargs: {})
    monkeypatch.setattr(core.legality, "LegalityChecker", lambda *args, **kwargs: object())

    calls = []

    def fake_completed(**kwargs):
        calls.append(kwargs)
        values = np.full(len(kwargs["date_indices"]), 0.25, dtype=np.float64)
        values[-1] = 0.5
        return values

    monkeypatch.setattr(
        core.trend_timing,
        "compute_dual_completed_trend_multipliers",
        fake_completed,
    )
    monkeypatch.setattr(
        core.trend_timing,
        "compute_dual_trend_multipliers",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("gap-derived path used")),
    )

    plan = object()
    plan_calls = []

    def fake_build_rebalance_day(**kwargs):
        plan_calls.append(kwargs)
        return plan

    monkeypatch.setattr(core.strategy, "build_rebalance_day", fake_build_rebalance_day)

    result = core.strategy.build_strategy_day(
        trade_date=trade_date,
        all_stocks=["000001"],
        individual_config={
            "weights": {"Factor": 1.0},
            "buy_n": 1,
            "sell_m": 1,
            "limit_up_protection": False,
            "rebalance": True,
            "trend_risk_overlay": {"enabled": True, "mode": "DUAL_COMPLETED"},
        },
        factor_classes=[],
        filter_factor_classes=[],
        positions={},
        sellable_volumes={},
        cash=100.0,
        position_multiplier=0.8,
    )

    assert len(calls) == 1
    assert calls[0]["date_indices"] == list(range(4, 25))
    assert calls[0]["valid_dates"] == [
        datetime(2024, 1, day) for day in range(5, 26)
    ]
    assert plan_calls[0]["position_multiplier"] == 0.4
    assert result.position_multiplier == 0.4
    assert result.plan is plan


def test_build_strategy_day_loads_factor_plus_timing_history_without_mocking_scores(
    monkeypatch,
):
    dates = np.datetime64("2024-01-01") + np.arange(40).astype("timedelta64[D]")
    rows = len(dates)
    data = {
        "trade_dates": dates,
        "stock_codes": np.array(["000001", "000002"]),
        "open": np.full((rows, 2), 10.0, dtype=np.float64),
        "close": np.full((rows, 2), 10.01, dtype=np.float64),
        "preClose": np.full((rows, 2), 10.0, dtype=np.float64),
        "st_mask": np.zeros((rows, 2), dtype=bool),
    }
    load_calls = []

    def fake_load_runtime_npz(requested_dates, max_lookback=None):
        load_calls.append((requested_dates, max_lookback))
        return data

    monkeypatch.setattr(core.runtime, "load_runtime_npz", fake_load_runtime_npz)
    monkeypatch.setattr(core.backtest, "_compute_list_dates", lambda *args, **kwargs: {})
    monkeypatch.setattr(core.legality, "LegalityChecker", _AllowAllChecker)

    plan = object()
    plan_calls = []

    def fake_build_rebalance_day(**kwargs):
        plan_calls.append(kwargs)
        return plan

    monkeypatch.setattr(core.strategy, "build_rebalance_day", fake_build_rebalance_day)
    trade_date = dates[-1].astype("datetime64[D]").item()
    result = core.strategy.build_strategy_day(
        trade_date=trade_date,
        all_stocks=["000001", "000002"],
        individual_config={
            "weights": {"_CompletedFactor": 1.0},
            "buy_n": 1,
            "sell_m": 1,
            "limit_up_protection": False,
            "rebalance": True,
            "trend_risk_overlay": {
                "enabled": True,
                "mode": "dual_completed",
                "strict_history": True,
                "strict_warmup_multiplier": 1.0,
                "momentum_window": 3,
                "ma_window": 5,
                "strategy_momentum_window": 3,
                "strategy_ma_window": 5,
            },
        },
        factor_classes=[_CompletedFactor],
        filter_factor_classes=[],
        positions={},
        sellable_volumes={},
        cash=100.0,
    )

    assert len(load_calls) == 1
    assert load_calls[0][1] == 20
    assert 0.0 <= result.position_multiplier <= 1.0
    assert np.isfinite(result.position_multiplier)
    assert plan_calls[0]["position_multiplier"] == result.position_multiplier
    assert result.plan is plan


def test_build_strategy_day_propagates_dual_completed_history_failure(monkeypatch):
    trade_date = date(2024, 1, 25)
    data, scored = _scored_history(trade_date)
    monkeypatch.setattr(core.runtime, "load_runtime_npz", lambda *args, **kwargs: data)
    monkeypatch.setattr(core.backtest, "_compute_factor_scores", lambda *args, **kwargs: scored)

    def fail_strictly(**kwargs):
        raise ValueError("strict completed strategy index produced invalid returns")

    monkeypatch.setattr(
        core.trend_timing,
        "compute_dual_completed_trend_multipliers",
        fail_strictly,
    )
    monkeypatch.setattr(
        core.trend_timing,
        "compute_dual_trend_multipliers",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("gap-derived path used")),
    )
    monkeypatch.setattr(
        core.strategy,
        "build_rebalance_day",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("failure was hidden")),
    )

    with pytest.raises(ValueError, match="strict completed strategy index"):
        core.strategy.build_strategy_day(
            trade_date=trade_date,
            all_stocks=["000001"],
            individual_config={
                "weights": {"Factor": 1.0},
                "buy_n": 1,
                "sell_m": 1,
                "limit_up_protection": False,
                "rebalance": True,
                "trend_risk_overlay": {"enabled": True, "mode": "dual_completed"},
            },
            factor_classes=[],
            filter_factor_classes=[],
            positions={},
            sellable_volumes={},
            cash=100.0,
        )
