import time
from datetime import datetime

import numpy as np
import pytest

from factor_db.factors.SameCalendarMonthReturn3YStrict import (
    SameCalendarMonthReturn3YStrict,
)


_FIRST_MONTH = np.datetime64("2014-01", "M")


def _panel(months=48, stocks=4, rows_per_month=2):
    dates = []
    month_numbers = []
    days_in_month = []
    for month_number in range(months):
        month_start = (_FIRST_MONTH + month_number).astype("datetime64[D]")
        for within_month_day in range(rows_per_month):
            dates.append(month_start + within_month_day)
            month_numbers.append(month_number)
            days_in_month.append(within_month_day)

    trade_dates = np.asarray(dates, dtype="datetime64[D]")
    month = np.asarray(month_numbers, dtype=np.float64)[:, None]
    within_month = np.asarray(days_in_month, dtype=np.float64)[:, None]
    stock = np.arange(stocks, dtype=np.float64)[None, :]
    pre_close = 8.0 + 0.01 * month + 0.3 * stock
    daily_return = (
        0.0007 * np.sin(month / 4.0 + within_month / 3.0)
        + 0.0001 * stock
    )
    close = pre_close * (1.0 + daily_return)
    open_ = np.full(close.shape, 5.0, dtype=np.float64)
    return {
        "trade_dates": trade_dates,
        "open": open_,
        "high": close * 1.02,
        "low": close * 0.98,
        "close": close,
        "volume": np.full(close.shape, 1_000_000.0, dtype=np.float64),
        "amount": np.full(close.shape, 10_000_000.0, dtype=np.float64),
        "preClose": pre_close,
        "st_mask": np.zeros(close.shape, dtype=bool),
    }


def _copy_panel(panel):
    return {name: values.copy() for name, values in panel.items()}


def _month_rows(panel, month_number):
    target = _FIRST_MONTH + month_number
    return np.flatnonzero(
        np.asarray(panel["trade_dates"]).astype("datetime64[M]") == target
    )


def _manual_score(panel, target_month, stock):
    monthly_returns = []
    months = np.asarray(panel["trade_dates"]).astype("datetime64[M]")
    for lag in (12, 24, 36):
        selected = months == (_FIRST_MONTH + target_month - lag)
        assert np.any(selected)
        gross = (
            panel["close"][selected, stock]
            / panel["preClose"][selected, stock]
        )
        monthly_returns.append(np.prod(gross, dtype=np.float64) - 1.0)
    return np.mean(monthly_returns, dtype=np.float64)


def test_three_completed_same_calendar_months_match_hand_calculation():
    panel = _panel(months=42, stocks=2)
    target_month = 40
    lag_gross = {
        12: np.array([[1.10, 0.90], [1.20, 1.00]]),
        24: np.array([[0.80, 1.25], [1.00, 0.80]]),
        36: np.array([[1.05, 1.10], [1.00, 1.10]]),
    }
    for lag, gross in lag_gross.items():
        rows = _month_rows(panel, target_month - lag)
        panel["close"][rows] = panel["preClose"][rows] * gross

    factor = SameCalendarMonthReturn3YStrict()
    result = factor.calc_batch(panel)
    target_rows = _month_rows(panel, target_month)

    expected = np.array(
        [
            ((1.10 * 1.20 - 1.0) + (0.80 * 1.00 - 1.0)
             + (1.05 * 1.00 - 1.0)) / 3.0,
            ((0.90 * 1.00 - 1.0) + (1.25 * 0.80 - 1.0)
             + (1.10 * 1.10 - 1.0)) / 3.0,
        ],
        dtype=np.float32,
    )
    assert factor.hist_days == 800
    assert result.shape == panel["close"].shape
    assert result.dtype == np.float32
    np.testing.assert_allclose(
        result[target_rows],
        np.broadcast_to(expected, (len(target_rows), 2)),
        rtol=1e-6,
        atol=0.0,
    )
    for stock in range(2):
        np.testing.assert_allclose(
            result[target_rows[0], stock],
            _manual_score(panel, target_month, stock),
            rtol=1e-6,
            atol=0.0,
        )


def test_first_boundary_month_is_not_assumed_complete_or_lag_shortened():
    factor = SameCalendarMonthReturn3YStrict()
    panel = _panel(months=37, stocks=2)
    assert np.isnan(factor.calc_batch(panel)).all()

    extended = _panel(months=38, stocks=2)
    first_usable = _month_rows(extended, 37)
    result = factor.calc_batch(extended)
    assert np.isnan(result[: first_usable[0]]).all()
    assert np.isfinite(result[first_usable]).all()


def test_uses_only_exact_12_24_and_36_month_boundaries():
    panel = _panel(months=43, stocks=1)
    factor = SameCalendarMonthReturn3YStrict()
    target_month = 40
    target_rows = _month_rows(panel, target_month)
    expected = factor.calc_batch(panel)[target_rows]

    for lag in (12, 24, 36):
        changed = _copy_panel(panel)
        row = _month_rows(panel, target_month - lag)[0]
        changed["close"][row, 0] *= 1.5
        assert factor.calc_batch(changed)[target_rows[0], 0] != expected[0, 0]

    for non_lag in (11, 13, 23, 25, 35, 37):
        changed = _copy_panel(panel)
        row = _month_rows(panel, target_month - non_lag)[0]
        changed["close"][row, 0] *= 1.5
        np.testing.assert_array_equal(
            factor.calc_batch(changed)[target_rows],
            expected,
        )


def test_equal_price_scaling_on_completed_days_is_corporate_action_invariant():
    panel = _panel(months=43, stocks=3)
    factor = SameCalendarMonthReturn3YStrict()
    target_rows = _month_rows(panel, 40)
    expected = factor.calc_batch(panel)[target_rows]

    scaled = _copy_panel(panel)
    completed_row = _month_rows(panel, 16)[1]
    scaled["close"][completed_row] *= 0.2
    scaled["preClose"][completed_row] *= 0.2

    np.testing.assert_allclose(
        factor.calc_batch(scaled)[target_rows],
        expected,
        rtol=1e-6,
        atol=0.0,
    )


def test_uses_official_daily_gross_not_raw_cross_day_close_ratios():
    panel = _panel(months=43, stocks=3)
    factor = SameCalendarMonthReturn3YStrict()
    expected = factor.calc_batch(panel)
    scaled = _copy_panel(panel)
    row_scale = np.exp(
        2.0 * np.sin(np.arange(len(panel["trade_dates"]), dtype=np.float64))
    )[:, None]
    scaled["close"] *= row_scale
    scaled["preClose"] *= row_scale

    np.testing.assert_allclose(
        factor.calc_batch(scaled),
        expected,
        rtol=1e-6,
        atol=1e-7,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "field",
    ("high", "low", "close", "volume", "amount", "preClose"),
)
def test_current_month_hlcva_and_preclose_cannot_change_frozen_signal(field):
    panel = _panel(months=43, stocks=4)
    factor = SameCalendarMonthReturn3YStrict()
    target_rows = _month_rows(panel, 40)
    expected = factor.calc_batch(panel)[target_rows]
    changed = _copy_panel(panel)
    changed[field][target_rows] = np.array(
        [np.nan, np.inf, -1.0, 1e30],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(
        factor.calc_batch(changed)[target_rows],
        expected,
    )


def test_new_historical_lag_takes_effect_only_at_next_month_boundary():
    panel = _panel(months=43, stocks=2)
    factor = SameCalendarMonthReturn3YStrict()
    current_month = 40
    current_rows = _month_rows(panel, current_month)
    next_rows = _month_rows(panel, current_month + 1)
    baseline = factor.calc_batch(panel)

    changed = _copy_panel(panel)
    # M-11 is not a lag for M, but it is exactly (M+1)-12.
    changed_row = _month_rows(panel, current_month - 11)[0]
    changed["close"][changed_row, 0] *= 1.5
    actual = factor.calc_batch(changed)

    np.testing.assert_array_equal(actual[current_rows], baseline[current_rows])
    assert actual[next_rows[0], 0] != baseline[next_rows[0], 0]
    np.testing.assert_array_equal(actual[next_rows, 1], baseline[next_rows, 1])


@pytest.mark.parametrize("field", ("close", "preClose"))
@pytest.mark.parametrize("value", (np.nan, np.inf, 0.0, -1.0))
def test_missing_or_invalid_row_poisons_only_exact_lags_then_recovers(
    field,
    value,
):
    panel = _panel(months=80, stocks=2)
    poisoned_month = 40
    poisoned_row = _month_rows(panel, poisoned_month)[0]
    panel[field][poisoned_row, 0] = value
    result = SameCalendarMonthReturn3YStrict().calc_batch(panel)
    poisoned_targets = {52, 64, 76}

    for target_month in range(37, 80):
        rows = _month_rows(panel, target_month)
        if target_month in poisoned_targets:
            assert np.isnan(result[rows, 0]).all()
            assert np.isfinite(result[rows, 1]).all()
        else:
            assert np.isfinite(result[rows]).all()

    for target_month in poisoned_targets:
        assert np.isfinite(result[_month_rows(panel, target_month + 1), 0]).all()


def test_current_open_and_st_are_per_row_legality_gates_only():
    panel = _panel(months=42, stocks=5)
    factor = SameCalendarMonthReturn3YStrict()
    target_row = _month_rows(panel, 40)[0]
    expected = factor.calc_batch(panel)[target_row]
    assert np.isfinite(expected).all()

    panel["open"][target_row] = np.array([2.0, 2000.0, 1.99, np.nan, np.inf])
    panel["st_mask"][target_row, 1] = True
    actual = factor.calc_batch(panel)[target_row]

    np.testing.assert_allclose(actual[0], expected[0])
    assert np.isnan(actual[1:]).all()


def test_exact_800_row_production_crop_preserves_target_month_score():
    # 21 rows/month makes the exact 800-row cut start inside a month.  The
    # factor must discard that partial boundary month while retaining the
    # complete M-12/M-24/M-36 inputs for the target month.
    panel = _panel(months=55, stocks=3, rows_per_month=21)
    factor = SameCalendarMonthReturn3YStrict()
    target_month = 54
    target_rows = _month_rows(panel, target_month)
    expected = factor.calc_batch(panel)[target_rows]

    crop_start = len(panel["trade_dates"]) - factor.hist_days
    cropped = {
        name: values[crop_start:].copy()
        for name, values in panel.items()
    }
    assert len(cropped["trade_dates"]) == 800
    cropped_target = np.flatnonzero(
        cropped["trade_dates"].astype("datetime64[M]")
        == (_FIRST_MONTH + target_month)
    )

    np.testing.assert_allclose(
        factor.calc_batch(cropped)[cropped_target],
        expected,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("field", ("open", "close", "preClose", "st_mask"))
def test_rejects_panel_shape_mismatch(field):
    panel = _panel()
    panel[field] = panel[field][:-1]
    with pytest.raises(ValueError, match="matching shapes"):
        SameCalendarMonthReturn3YStrict().calc_batch(panel)


@pytest.mark.parametrize("field", ("open", "close", "preClose"))
def test_rejects_non_matrix_or_non_real_numeric_prices(field):
    panel = _panel()
    panel[field] = panel[field][:, 0]
    with pytest.raises(ValueError, match="two-dimensional"):
        SameCalendarMonthReturn3YStrict().calc_batch(panel)

    panel = _panel()
    panel[field] = panel[field].astype(str)
    with pytest.raises(ValueError, match="real numeric dtype"):
        SameCalendarMonthReturn3YStrict().calc_batch(panel)


def test_rejects_non_boolean_st_mask():
    panel = _panel()
    panel["st_mask"] = panel["st_mask"].astype(np.int8)
    with pytest.raises(ValueError, match="boolean"):
        SameCalendarMonthReturn3YStrict().calc_batch(panel)


def test_accepts_python_date_and_datetime_trade_dates():
    panel = _panel()
    factor = SameCalendarMonthReturn3YStrict()
    expected = factor.calc_batch(panel)

    date_panel = _copy_panel(panel)
    date_panel["trade_dates"] = [
        value.astype("datetime64[D]").item()
        for value in panel["trade_dates"]
    ]
    np.testing.assert_array_equal(factor.calc_batch(date_panel), expected)

    datetime_panel = _copy_panel(panel)
    datetime_panel["trade_dates"] = [
        datetime.combine(value, datetime.min.time())
        for value in date_panel["trade_dates"]
    ]
    np.testing.assert_array_equal(factor.calc_batch(datetime_panel), expected)


def test_rejects_invalid_trade_date_shape_length_dtype_string_and_nat():
    factor = SameCalendarMonthReturn3YStrict()

    panel = _panel()
    panel["trade_dates"] = panel["trade_dates"][:-1]
    with pytest.raises(ValueError, match="length"):
        factor.calc_batch(panel)

    panel = _panel()
    panel["trade_dates"] = panel["trade_dates"][:, None]
    with pytest.raises(ValueError, match="one-dimensional"):
        factor.calc_batch(panel)

    panel = _panel()
    panel["trade_dates"] = np.arange(len(panel["trade_dates"]), dtype=np.int64)
    with pytest.raises(ValueError, match="datetime64 or date-like"):
        factor.calc_batch(panel)

    panel = _panel()
    panel["trade_dates"] = panel["trade_dates"].astype(object)
    panel["trade_dates"][4] = "not-a-date"
    with pytest.raises(ValueError, match="valid calendar dates"):
        factor.calc_batch(panel)

    panel = _panel()
    panel["trade_dates"] = panel["trade_dates"].copy()
    panel["trade_dates"][4] = np.datetime64("NaT")
    with pytest.raises(ValueError, match="NaT"):
        factor.calc_batch(panel)


@pytest.mark.parametrize("mode", ("duplicate", "reverse"))
def test_rejects_non_monotonic_trade_dates(mode):
    panel = _panel()
    panel["trade_dates"] = panel["trade_dates"].copy()
    if mode == "duplicate":
        panel["trade_dates"][3] = panel["trade_dates"][2]
    else:
        panel["trade_dates"][[2, 3]] = panel["trade_dates"][[3, 2]]
    with pytest.raises(ValueError, match="strictly increasing"):
        SameCalendarMonthReturn3YStrict().calc_batch(panel)


def test_rejects_missing_natural_month_in_trade_dates():
    panel = _panel()
    omitted_month = _FIRST_MONTH + 10
    keep = panel["trade_dates"].astype("datetime64[M]") != omitted_month
    original_rows = len(keep)
    for name, values in list(panel.items()):
        if values.ndim >= 1 and len(values) == original_rows:
            panel[name] = values[keep]

    with pytest.raises(ValueError, match="consecutive calendar months"):
        SameCalendarMonthReturn3YStrict().calc_batch(panel)


@pytest.mark.parametrize(("rows", "stocks"), ((0, 3), (2, 0)))
def test_rejects_empty_panel_dimensions(rows, stocks):
    panel = _panel(months=1, stocks=stocks, rows_per_month=rows)
    with pytest.raises(ValueError, match="at least one row and stock"):
        SameCalendarMonthReturn3YStrict().calc_batch(panel)


def test_800_by_1500_panel_is_vectorized_across_stocks():
    rows_per_month = 20
    panel = _panel(months=40, stocks=1500, rows_per_month=rows_per_month)
    started = time.perf_counter()
    result = SameCalendarMonthReturn3YStrict().calc_batch(panel)
    elapsed = time.perf_counter() - started

    first_usable = _month_rows(panel, 37)[0]
    assert result.shape == (800, 1500)
    assert result.dtype == np.float32
    assert np.isnan(result[:first_usable]).all()
    assert np.isfinite(result[first_usable:]).all()
    assert elapsed < 1.0
