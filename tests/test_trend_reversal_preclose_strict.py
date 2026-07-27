import numpy as np

from factor_db.factors.TrendReversalPreCloseStrict import (
    TrendReversalPreCloseStrict,
)
from factor_db.factors.TrendOpenSignal import _lag, _rolling_mean, _rolling_sum
from factor_db.factors.TrendOpenSignalV4 import _rolling_extreme


def _reference_rolling_max_complete(values, window):
    finite_count = _rolling_sum(np.isfinite(values).astype(np.float64), window)
    maximum = _rolling_extreme(values, window, "max")
    return np.where(finite_count == window, maximum, np.nan)


def _reference_calc(panel):
    """Original full-matrix implementation retained as an exact oracle."""
    open_ = np.asarray(panel["open"], dtype=np.float64)
    high = np.asarray(panel["high"], dtype=np.float64)
    close = np.asarray(panel["close"], dtype=np.float64)
    pre_close = np.asarray(panel["preClose"], dtype=np.float64)
    st_mask = np.asarray(panel["st_mask"], dtype=bool)

    valid_close = (
        np.isfinite(close)
        & (close > 0.0)
        & np.isfinite(pre_close)
        & (pre_close > 0.0)
    )
    valid_high = valid_close & np.isfinite(high) & (high > 0.0) & (high >= close)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        close_gross = np.where(valid_close, close / pre_close, np.nan)
        daily_return = close_gross - 1.0
        log_close_gross = np.where(valid_close, np.log(close_gross), np.nan)
        log_high_gross = np.where(valid_high, np.log(high / pre_close), np.nan)

    completed_log_return = _lag(log_close_gross, 1)
    completed_daily_return = _lag(daily_return, 1)
    with np.errstate(over="ignore", invalid="ignore"):
        ret20 = np.expm1(_rolling_sum(completed_log_return, 20))
        ret60 = np.expm1(_rolling_sum(completed_log_return, 60))
        volatility20 = np.sqrt(
            _rolling_mean(completed_daily_return * completed_daily_return, 20)
        )

    filled_log_return = np.where(valid_close, log_close_gross, 0.0)
    cumulative_log_close = np.cumsum(filled_log_return, axis=0, dtype=np.float64)
    log_wealth_before_day = cumulative_log_close - filled_log_return
    adjusted_log_high = np.where(
        valid_high,
        log_wealth_before_day + log_high_gross,
        np.nan,
    )
    completed_adjusted_high = _lag(adjusted_log_high, 1)
    peak20 = _reference_rolling_max_complete(completed_adjusted_high, 20)
    completed_log_wealth = _lag(cumulative_log_close, 1)
    with np.errstate(over="ignore", invalid="ignore"):
        drawdown20 = np.expm1(completed_log_wealth - peak20)

    components_finite = (
        np.isfinite(ret20)
        & np.isfinite(ret60)
        & np.isfinite(volatility20)
        & np.isfinite(drawdown20)
    )
    score = (
        0.45 * np.clip((-ret20 + 0.20) / 0.40, 0.0, 1.0)
        + 0.25 * np.clip((-ret60 + 0.40) / 0.80, 0.0, 1.0)
        + 0.15 * np.clip((0.08 - volatility20) / 0.08, 0.0, 1.0)
        + 0.15 * np.clip((drawdown20 + 0.35) / 0.35, 0.0, 1.0)
    )
    legal = np.isfinite(open_) & (open_ >= 2.0) & ~st_mask
    valid = legal & components_finite & np.isfinite(score)
    return np.where(valid, score, np.nan).astype(np.float32, copy=False)


def _panel(rows=72, stocks=4):
    time = np.arange(rows, dtype=np.float64)[:, None]
    stock = np.arange(stocks, dtype=np.float64)[None, :]
    pre_close = 10.0 + 0.03 * time + stock
    daily_return = 0.002 * np.sin(time / 7.0) + 0.0002 * stock
    close = pre_close * (1.0 + daily_return)
    return {
        "open": close + 0.01,
        "high": np.maximum(close, pre_close * 1.01),
        "close": close,
        "preClose": pre_close,
        "st_mask": np.zeros((rows, stocks), dtype=bool),
    }


def test_exposes_incomplete_and_non_finite_official_return_history():
    panel = _panel()
    panel["preClose"][10, 1] = np.nan
    panel["high"][50, 2] = np.inf
    result = TrendReversalPreCloseStrict().calc_batch(panel)

    assert np.isnan(result[:60]).all()
    assert np.isfinite(result[60, 0])
    assert np.isnan(result[60, 1])
    assert np.isnan(result[60, 2])
    assert np.isfinite(result[60, 3])


def test_current_day_hlc_and_preclose_cannot_change_current_score():
    panel = _panel()
    factor = TrendReversalPreCloseStrict()
    expected = factor.calc_batch(panel)[-1]
    changed = {key: value.copy() for key, value in panel.items()}
    for name in ("high", "close", "preClose"):
        changed[name][-1] = np.array([np.nan, np.inf, -1.0, 1e30])

    np.testing.assert_array_equal(factor.calc_batch(changed)[-1], expected)


def test_completed_corporate_action_scale_is_invariant():
    panel = _panel()
    factor = TrendReversalPreCloseStrict()
    expected = factor.calc_batch(panel)[-1]

    scaled = {key: value.copy() for key, value in panel.items()}
    scaled["close"][30] *= 0.2
    scaled["high"][30] *= 0.2
    scaled["preClose"][30] *= 0.2
    np.testing.assert_allclose(factor.calc_batch(scaled)[-1], expected, rtol=1e-6)


def test_current_open_and_st_are_legality_gates_only():
    panel = _panel()
    factor = TrendReversalPreCloseStrict()
    expected = factor.calc_batch(panel)[-1]
    assert np.isfinite(expected).all()

    panel["open"][-1] = np.array([2.0, 2000.0, 1.99, np.nan])
    panel["st_mask"][-1, 1] = True
    actual = factor.calc_batch(panel)[-1]

    np.testing.assert_allclose(actual[0], expected[0])
    assert np.isnan(actual[1:]).all()


def test_matches_exact_reference_with_random_missing_and_corporate_actions():
    rng = np.random.default_rng(20260722)
    rows, stocks = 143, 11
    pre_close = rng.uniform(2.5, 80.0, size=(rows, stocks))
    daily_return = rng.normal(0.0005, 0.025, size=(rows, stocks))
    close = pre_close * (1.0 + daily_return)
    high = np.maximum(close, pre_close) * rng.uniform(
        1.0,
        1.035,
        size=(rows, stocks),
    )
    panel = {
        "open": pre_close * rng.uniform(0.96, 1.04, size=(rows, stocks)),
        "high": high,
        "close": close,
        "preClose": pre_close,
        "st_mask": rng.random((rows, stocks)) < 0.03,
    }

    # Completed-day corporate actions rescale all official price fields but do
    # not alter either official returns or adjusted-high drawdown geometry.
    for day, scale in ((17, 0.2), (51, 1.7), (96, 0.35)):
        for field in ("open", "high", "close", "preClose"):
            panel[field][day] *= scale

    locations = rng.choice(rows * stocks, size=34, replace=False)
    row_index, stock_index = np.unravel_index(locations, (rows, stocks))
    panel["preClose"][row_index[:8], stock_index[:8]] = np.nan
    panel["close"][row_index[8:16], stock_index[8:16]] = np.nan
    panel["high"][row_index[16:22], stock_index[16:22]] = np.inf
    panel["high"][row_index[22:28], stock_index[22:28]] = 0.0
    panel["open"][row_index[28:31], stock_index[28:31]] = np.nan
    panel["open"][row_index[31:], stock_index[31:]] = 1.99

    expected = _reference_calc(panel)
    actual = TrendReversalPreCloseStrict().calc_batch(panel)

    np.testing.assert_array_equal(actual, expected)
