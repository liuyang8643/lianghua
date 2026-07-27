import numpy as np
import pytest
from pathlib import Path
from datetime import datetime, timedelta

from core.trend_timing import (
    _apply_recovery_limit,
    build_strategy_topn_path,
    compute_configured_timing_multipliers,
    compute_dual_completed_trend_multipliers,
    compute_dual_trend_multipliers,
    dual_trend_multipliers,
    market_open_index,
    market_completed_index,
    strategy_completed_index,
    strategy_open_index,
    strategy_trend_multipliers,
    trend_overlay_multipliers,
)


def _data(days=80, stocks=4):
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0, 0.01, size=(days, stocks))
    close = np.cumprod(1.0 + returns, axis=0) * 10.0
    pre_close = np.empty_like(close)
    pre_close[0] = close[0]
    pre_close[1:] = close[:-1]
    open_price = close.copy()
    return {
        "open": open_price,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "preClose": pre_close,
        "volume": np.ones_like(close),
        "amount": np.ones_like(close),
        "st_mask": np.zeros((days, stocks), dtype=bool),
    }


def _config(rule="or"):
    return {"trend_risk_overlay": {
        "enabled": True,
        "risk_rule": rule,
        "momentum_window": 5,
        "momentum_floor": -0.05,
        "ma_window": 20,
        "ma_ratio_floor": 1.0,
        "risk_floor": 0.3,
    }}


def test_overlay_current_row_is_causal():
    data = _data()
    expected = trend_overlay_multipliers(data, _config())
    changed = {key: value.copy() for key, value in data.items()}
    for key in ("high", "low", "close", "volume", "amount"):
        changed[key][50] *= 100.0
    actual = trend_overlay_multipliers(changed, _config())
    np.testing.assert_array_equal(actual[50], expected[50])

    changed_open = {key: value.copy() for key, value in data.items()}
    changed_open["open"][50] *= 0.8
    actual_open = trend_overlay_multipliers(changed_open, _config())
    assert actual_open[50] != expected[50]


def test_overlay_rejects_unknown_rule():
    with pytest.raises(ValueError, match="risk_rule"):
        trend_overlay_multipliers(_data(), _config("xor"))


def test_overlay_recovery_is_gradual_but_derisking_is_immediate():
    target = np.array([1.0, 0.08, 1.0, 1.0, 0.08])

    actual = _apply_recovery_limit(target, recovery_step=0.9)

    np.testing.assert_allclose(actual, [1.0, 0.08, 0.98, 1.0, 0.08])


def test_overlay_rejects_invalid_recovery_step():
    config = _config()
    config["trend_risk_overlay"]["recovery_step"] = 0.0

    with pytest.raises(ValueError, match="recovery_step"):
        trend_overlay_multipliers(_data(), config)


def test_continuous_overlay_is_causal_and_not_bucketed():
    config = {"trend_risk_overlay": {
        "enabled": True, "mode": "continuous_score", "floor": 0.05,
        "momentum_window": 5, "ma_window": 20,
        "momentum_center": -0.05, "momentum_scale": 0.004,
        "ma_center": 1.0, "ma_scale": 0.012,
        "softmin_sharpness": 4.0, "slope": 2.0,
    }}
    data = _data(days=160, stocks=12)
    expected = trend_overlay_multipliers(data, config)
    changed = {key: value.copy() for key, value in data.items()}
    changed["close"][100] *= 50.0
    actual = trend_overlay_multipliers(changed, config)

    assert actual[100] == expected[100]
    assert np.unique(np.round(expected[30:], 6)).size > 50
    assert np.all((expected >= 0.05) & (expected <= 1.0))


def test_strategy_and_dual_trend_are_causal_continuous_values():
    settings = {
        "floor": 0.03, "momentum_window": 5, "ma_window": 20,
        "momentum_center": -0.055, "momentum_scale": 0.012,
        "ma_center": 1.0, "ma_scale": 0.012,
        "softmin_sharpness": 4.0, "slope": 2.0,
        "strategy_weight": 0.6,
    }
    open_index = np.cumprod(1.0 + np.sin(np.arange(180) / 4.0) * 0.015)
    expected = strategy_trend_multipliers(open_index, settings)
    changed = open_index.copy()
    changed[100] *= 0.8
    actual = strategy_trend_multipliers(changed, settings)
    assert actual[100] != expected[100]
    assert np.unique(np.round(expected[30:], 6)).size > 80

    market_index = market_open_index(_data(days=180, stocks=12))
    dual = dual_trend_multipliers(market_index, open_index, settings)
    assert np.unique(np.round(dual[30:], 6)).size > 100
    assert np.all((dual >= 0.03) & (dual <= 1.0))


def test_dual_trend_may_use_current_open_but_not_current_hlc_or_future_rows():
    data = _data(days=180, stocks=12)
    strategy_index = np.cumprod(1.0 + np.sin(np.arange(180) / 4.0) * 0.015)
    settings = {
        "floor": 0.03, "ceiling": 1.0,
        "momentum_window": 5, "ma_window": 20,
        "momentum_center": -0.055, "momentum_scale": 0.012,
        "ma_center": 1.0, "ma_scale": 0.012,
        "softmin_sharpness": 4.0, "slope": 2.0,
        "strategy_weight": 0.6,
    }
    expected = dual_trend_multipliers(
        market_open_index(data), strategy_index, settings,
    )

    changed_open = {key: value.copy() for key, value in data.items()}
    changed_open["open"][100] *= 0.8
    actual_open = dual_trend_multipliers(
        market_open_index(changed_open), strategy_index, settings,
    )
    assert actual_open[100] != expected[100]

    changed_forbidden = {key: value.copy() for key, value in data.items()}
    for key in ("high", "low", "close", "volume", "amount"):
        changed_forbidden[key][100] *= 50.0
    actual_forbidden = dual_trend_multipliers(
        market_open_index(changed_forbidden), strategy_index, settings,
    )
    assert actual_forbidden[100] == expected[100]

    changed_future = {key: value.copy() for key, value in data.items()}
    changed_future["open"][101:] *= 1.10
    actual_future = dual_trend_multipliers(
        market_open_index(changed_future), strategy_index, settings,
    )
    np.testing.assert_array_equal(actual_future[:101], expected[:101])


def test_strategy_open_index_uses_previous_day_targets():
    data = _data(days=3, stocks=2)
    data["open"][:] = 10.0
    data["close"][:] = 10.0
    data["preClose"][:] = 10.0
    data["open"][1, 0] *= 1.10
    data["open"][1, 1] *= 0.90
    daily_topn = [["A"], ["B"], ["A"]]

    actual = strategy_open_index(
        data, daily_topn, [0, 1, 2], {"A": 0, "B": 1},
    )

    assert actual[0] == 1.0
    assert actual[1] == pytest.approx(1.10)


def test_completed_market_index_uses_only_official_returns_through_t_minus_1():
    data = _data(days=80, stocks=4)
    expected = market_completed_index(data)

    current = {key: value.copy() for key, value in data.items()}
    current["close"][50] *= 1.20
    current["high"][50] *= 2.0
    current["low"][50] *= 0.5
    current["open"][50] *= 1.5
    actual = market_completed_index(current)
    assert actual[50] == expected[50]
    assert actual[51] != expected[51]

    scaled = {key: value.copy() for key, value in data.items()}
    scaled["close"][49] *= 0.25
    scaled["preClose"][49] *= 0.25
    np.testing.assert_allclose(market_completed_index(scaled), expected)


def test_completed_strategy_index_shifts_same_day_target_return_to_next_open():
    data = _data(days=3, stocks=2)
    data["close"][:] = 10.0
    data["preClose"][:] = 10.0
    data["close"][0, 0] = 11.0
    data["close"][0, 1] = 9.0
    targets = [["A"], ["B"], ["A"]]

    actual = strategy_completed_index(
        data, targets, [0, 1, 2], {"A": 0, "B": 1},
    )

    np.testing.assert_allclose(actual, [1.0, 1.10, 1.10])


def test_dual_completed_multiplier_is_strict_and_has_no_current_hlcva_input():
    days = 80
    code = "000001.SZ"
    start = datetime(2024, 1, 2)
    dates = [start + timedelta(days=i) for i in range(days)]
    data = _data(days=days, stocks=1)
    data["stock_codes"] = np.array([code])
    data["issue_price"] = np.array([10.0])
    data["trade_dates"] = np.array(
        [d.date() for d in dates], dtype="datetime64[D]",
    )
    settings = {
        "floor": 0.0, "ceiling": 1.0,
        "momentum_window": 5, "ma_window": 20,
        "momentum_center": -0.055, "momentum_scale": 0.012,
        "ma_center": 1.01, "ma_scale": 0.009,
        "softmin_sharpness": 4.0, "slope": 2.0,
        "strategy_weight": 0.8,
        "strategy_momentum_window": 5,
        "strategy_momentum_center": -0.044,
        "strategy_momentum_scale": 0.015,
        "strategy_ma_window": 20,
        "strategy_ma_center": 1.014,
        "strategy_ma_scale": 0.009,
        "strategy_softmin_sharpness": 4.0,
        "strategy_slope": 2.0,
        "strict_history": True,
        "strict_warmup_multiplier": 1.0,
    }
    scores = {"Trend": np.ones((days, 1), dtype=np.float32)}
    expected = compute_dual_completed_trend_multipliers(
        data=data, all_scores=scores, valid_dates=dates,
        date_indices=list(range(days)), valid_stocks=[code],
        stock_indices={code: 0}, weights={"Trend": 1.0}, buy_n=1,
        settings=settings,
    )

    changed = {key: value.copy() for key, value in data.items()}
    for key in ("high", "low", "close", "volume", "amount"):
        changed[key][50] *= 10.0
    actual = compute_dual_completed_trend_multipliers(
        data=changed, all_scores=scores, valid_dates=dates,
        date_indices=list(range(days)), valid_stocks=[code],
        stock_indices={code: 0}, weights={"Trend": 1.0}, buy_n=1,
        settings=settings,
    )

    assert actual[50] == expected[50]
    assert actual[51] != expected[51]


def test_generic_backtest_has_no_strategy_timing_coupling():
    source = Path("core/backtest.py").read_text(encoding="utf-8")
    assert "dual_shadow" not in source
    assert "dual_strategy" not in source
    assert "trend_timing" not in source
    timing_source = Path("core/trend_timing.py").read_text(encoding="utf-8")
    assert "_backtest_direct" not in timing_source
    assert "core.backtest" not in timing_source


def test_target_path_matches_generic_backtest_topn(monkeypatch):
    from core.backtest import _backtest_direct

    days = 8
    codes = ["000001.SZ", "000002.SZ"]
    start = datetime(2024, 1, 2)
    dates = [start + timedelta(days=i) for i in range(days)]
    data = {
        "stock_codes": np.array(codes),
        "trade_dates": np.array([d.date() for d in dates], dtype="datetime64[D]"),
        "open": np.full((days, 2), 10.0),
        "high": np.full((days, 2), 10.1),
        "low": np.full((days, 2), 9.9),
        "close": np.full((days, 2), 10.0),
        "volume": np.full((days, 2), 1_000_000.0),
        "amount": np.full((days, 2), 10_000_000.0),
        "preClose": np.full((days, 2), 10.0),
        "st_mask": np.zeros((days, 2), dtype=bool),
        "issue_price": np.array([10.0, 10.0]),
    }
    scores = np.array([
        [1.0, 0.5] if i % 2 == 0 else [0.5, 1.0]
        for i in range(days)
    ], dtype=np.float32)
    all_scores = {"Trend": scores}
    stock_indices = {code: i for i, code in enumerate(codes)}
    targets = build_strategy_topn_path(
        data=data, all_scores=all_scores, valid_dates=dates,
        date_indices=list(range(days)), valid_stocks=codes,
        stock_indices=stock_indices, weights={"Trend": 1.0}, buy_n=1,
    )
    monkeypatch.setattr("core.backtest.get_delist_stock_info", lambda: {})
    shadow = _backtest_direct(
        data, all_scores, dates, list(range(days)), codes, stock_indices,
        weights={"Trend": 1.0}, buy_n=1, sell_m=1,
        lightweight=True, market_order_freeze=False, list_dates_map={},
    )

    assert targets == shadow["daily_topn"]


def test_target_path_timing_does_not_run_a_shadow_account():
    days = 35
    code = "000001.SZ"
    start = datetime(2024, 1, 2)
    dates = [start + timedelta(days=i) for i in range(days)]
    data = {
        "stock_codes": np.array([code]),
        "trade_dates": np.array([d.date() for d in dates], dtype="datetime64[D]"),
        "open": np.full((days, 1), 10.0),
        "high": np.full((days, 1), 10.1),
        "low": np.full((days, 1), 9.9),
        "close": np.full((days, 1), 10.0),
        "volume": np.full((days, 1), 1_000_000.0),
        "amount": np.full((days, 1), 10_000_000.0),
        "preClose": np.full((days, 1), 10.0),
        "st_mask": np.zeros((days, 1), dtype=bool),
        "issue_price": np.array([10.0]),
    }
    actual = compute_dual_trend_multipliers(
        data=data,
        all_scores={"Trend": np.ones((days, 1), dtype=np.float32)},
        valid_dates=dates, date_indices=list(range(days)),
        valid_stocks=[code], stock_indices={code: 0},
        weights={"Trend": 1.0}, buy_n=1,
        settings={
            "floor": 0.03, "ceiling": 1.0,
            "momentum_window": 3, "ma_window": 10,
            "momentum_center": -0.05, "momentum_scale": 0.01,
            "ma_center": 1.0, "ma_scale": 0.01,
            "softmin_sharpness": 4.0, "slope": 2.0,
            "strategy_weight": 0.6,
        },
    )

    assert actual.shape == (days,)
    assert np.all((actual >= 0.03) & (actual <= 1.0))


def test_configured_timing_combines_dual_overlay_with_base_multiplier():
    days = 35
    code = "000001.SZ"
    start = datetime(2024, 1, 2)
    dates = [start + timedelta(days=i) for i in range(days)]
    data = _data(days=days, stocks=1)
    data["stock_codes"] = np.array([code])
    data["issue_price"] = np.array([10.0])
    data["trade_dates"] = np.array(
        [d.date() for d in dates], dtype="datetime64[D]",
    )
    config = {
        "weights": {"Trend": 1.0}, "buy_n": 1,
        "limit_up_protection": False,
        "trend_risk_overlay": {
            "enabled": True, "mode": "dual_strategy",
            "floor": 0.0, "ceiling": 1.0,
            "momentum_window": 3, "ma_window": 10,
            "momentum_center": -0.05, "momentum_scale": 0.01,
            "ma_center": 1.0, "ma_scale": 0.01,
            "softmin_sharpness": 4.0, "slope": 2.0,
            "strategy_weight": 1.0,
            "strategy_momentum_window": 3,
            "strategy_momentum_center": -0.05,
            "strategy_momentum_scale": 0.01,
            "strategy_ma_window": 10,
            "strategy_ma_center": 1.0,
            "strategy_ma_scale": 0.01,
            "strategy_softmin_sharpness": 4.0,
            "strategy_slope": 2.0,
        },
    }
    actual = compute_configured_timing_multipliers(
        data=data,
        all_scores={"Trend": np.ones((days, 1), dtype=np.float32)},
        valid_dates=dates, date_indices=list(range(days)),
        valid_stocks=[code], stock_indices={code: 0}, config=config,
        base_multipliers=np.full(days, 0.5),
    )

    assert actual is not None
    assert np.all((actual >= 0.0) & (actual <= 0.5))
