import time
from datetime import datetime

import numpy as np
import pytest

from factor_db.factors.Completed52WeekHighProximityStrict import (
    Completed52WeekHighProximityStrict,
)


def _business_dates(rows):
    calendar = np.arange(
        np.datetime64("2018-01-01"),
        np.datetime64("2030-01-01"),
        dtype="datetime64[D]",
    )
    dates = calendar[np.is_busday(calendar)]
    assert len(dates) >= rows
    return dates[:rows]


def _panel(rows=420, stocks=4):
    trade_dates = _business_dates(rows)
    day = np.arange(rows, dtype=np.float64)[:, None]
    stock = np.arange(stocks, dtype=np.float64)[None, :]
    pre_close = 6.0 + 0.11 * (day % 23.0) + 0.7 * stock
    gross = 1.0 + 0.003 * np.sin(day / 11.0 + stock / 7.0)
    close = pre_close * gross
    open_ = 5.0 + np.broadcast_to(0.01 * stock, close.shape)
    return {
        "trade_dates": trade_dates,
        "open": open_,
        "high": close * 1.02,
        "low": close * 0.98,
        "close": close,
        "volume": np.broadcast_to(1_000_000.0 + 100.0 * day, close.shape).copy(),
        "amount": np.broadcast_to(10_000_000.0 + 1_000.0 * day, close.shape).copy(),
        "preClose": pre_close,
        "st_mask": np.zeros(close.shape, dtype=bool),
    }


def _month_starts(panel):
    months = panel["trade_dates"].astype("datetime64[M]")
    return np.flatnonzero(np.r_[True, months[1:] != months[:-1]])


def _month_end(panel, start):
    starts = _month_starts(panel)
    index = int(np.flatnonzero(starts == start)[0])
    return (
        int(starts[index + 1])
        if index + 1 < len(starts)
        else len(panel["trade_dates"])
    )


def _eligible_month_starts(panel):
    return _month_starts(panel)[_month_starts(panel) >= 252]


def _manual_score(panel, month_start, stock):
    gross = (
        panel["close"][month_start - 252 : month_start, stock]
        / panel["preClose"][month_start - 252 : month_start, stock]
    )
    wealth_path = np.cumprod(gross, dtype=np.float64)
    return wealth_path[-1] / max(1.0, float(np.max(wealth_path)))


def _copy_panel(panel):
    return {name: values.copy() for name, values in panel.items()}


def test_exact_252_completed_days_matches_hand_calculation():
    panel = _panel(rows=420, stocks=2)
    factor = Completed52WeekHighProximityStrict()
    month_start = int(_eligible_month_starts(panel)[0])
    month_end = _month_end(panel, month_start)

    gross = np.ones((252, 2), dtype=np.float64)
    gross[0] = np.array([1.2, 0.8])
    gross[-1] = np.array([0.75, 1.125])
    panel["close"][month_start - 252 : month_start] = (
        panel["preClose"][month_start - 252 : month_start] * gross
    )

    result = factor.calc_batch(panel)

    assert factor.hist_days == 280
    assert result.shape == panel["close"].shape
    assert result.dtype == np.float32
    assert np.isnan(result[:month_start]).all()
    np.testing.assert_allclose(
        result[month_start:month_end],
        np.broadcast_to(
            np.array([0.75, 0.9], dtype=np.float32),
            (month_end - month_start, 2),
        ),
        rtol=1e-6,
        atol=0.0,
    )
    for stock in range(2):
        np.testing.assert_allclose(
            result[month_start, stock],
            _manual_score(panel, month_start, stock),
            rtol=1e-6,
            atol=0.0,
        )


def test_window_is_exactly_s_minus_252_through_s_exclusive():
    panel = _panel(rows=460, stocks=1)
    factor = Completed52WeekHighProximityStrict()
    month_start = int(_eligible_month_starts(panel)[1])
    month_end = _month_end(panel, month_start)
    expected = factor.calc_batch(panel)[month_start:month_end]

    outside = _copy_panel(panel)
    outside["close"][month_start - 253, 0] = np.nan
    np.testing.assert_array_equal(
        factor.calc_batch(outside)[month_start:month_end],
        expected,
    )

    inside = _copy_panel(panel)
    inside["close"][month_start - 252, 0] = np.nan
    assert np.isnan(factor.calc_batch(inside)[month_start:month_end, 0]).all()


def test_280_row_loader_buffer_reaches_month_start_after_23_row_month():
    panel = _panel(rows=650, stocks=1)
    starts = _month_starts(panel)
    selected = None
    for month_start in starts:
        month_start = int(month_start)
        month_end = _month_end(panel, month_start)
        if month_end - month_start == 23 and month_end >= 280:
            selected = (month_start, month_end)
            break
    assert selected is not None
    month_start, month_end = selected

    full = Completed52WeekHighProximityStrict().calc_batch(panel)
    crop_start = month_end - 280
    cropped = {
        name: values[crop_start:month_end].copy()
        for name, values in panel.items()
    }
    local_month_start = month_start - crop_start
    assert local_month_start >= 252

    cropped_result = Completed52WeekHighProximityStrict().calc_batch(cropped)
    np.testing.assert_allclose(
        cropped_result[local_month_start:],
        full[month_start:month_end],
        rtol=0.0,
        atol=0.0,
    )


def test_uses_official_gross_returns_not_raw_cross_day_close_ratios():
    panel = _panel(rows=440, stocks=3)
    factor = Completed52WeekHighProximityStrict()
    expected = factor.calc_batch(panel)
    scaled = _copy_panel(panel)
    scale = np.exp(
        2.0 * np.sin(np.arange(len(panel["trade_dates"]), dtype=np.float64))
    )[:, None]
    scaled["close"] *= scale
    scaled["preClose"] *= scale

    np.testing.assert_allclose(
        factor.calc_batch(scaled),
        expected,
        rtol=1e-6,
        atol=1e-7,
        equal_nan=True,
    )


def test_equal_completed_day_scaling_is_corporate_action_invariant():
    panel = _panel(rows=440, stocks=3)
    factor = Completed52WeekHighProximityStrict()
    month_start = int(_eligible_month_starts(panel)[1])
    month_end = _month_end(panel, month_start)
    expected = factor.calc_batch(panel)[month_start:month_end]

    scaled = _copy_panel(panel)
    completed_day = month_start - 117
    scaled["close"][completed_day] *= 0.2
    scaled["preClose"][completed_day] *= 0.2

    np.testing.assert_allclose(
        factor.calc_batch(scaled)[month_start:month_end],
        expected,
        rtol=1e-6,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "field",
    ("high", "low", "close", "volume", "amount", "preClose"),
)
def test_current_month_hlcva_and_preclose_cannot_change_its_signal(field):
    panel = _panel(rows=440, stocks=4)
    factor = Completed52WeekHighProximityStrict()
    month_start = int(_eligible_month_starts(panel)[1])
    month_end = _month_end(panel, month_start)
    expected = factor.calc_batch(panel)[month_start:month_end]
    changed = _copy_panel(panel)
    changed[field][month_start:month_end] = np.array(
        [np.nan, np.inf, -1.0, 1e30],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(
        factor.calc_batch(changed)[month_start:month_end],
        expected,
    )


def test_completed_current_month_first_affects_the_next_month_boundary():
    panel = _panel(rows=460, stocks=2)
    factor = Completed52WeekHighProximityStrict()
    starts = _eligible_month_starts(panel)
    month_start = int(starts[1])
    month_end = _month_end(panel, month_start)
    next_start = int(starts[2])
    next_end = _month_end(panel, next_start)
    baseline = factor.calc_batch(panel)

    changed = _copy_panel(panel)
    changed["close"][month_start, 0] = np.nan
    actual = factor.calc_batch(changed)

    np.testing.assert_array_equal(
        actual[month_start:month_end],
        baseline[month_start:month_end],
    )
    assert np.isnan(actual[next_start:next_end, 0]).all()
    assert np.isfinite(actual[next_start:next_end, 1]).all()


@pytest.mark.parametrize("field", ("close", "preClose"))
@pytest.mark.parametrize("value", (np.nan, np.inf, 0.0, -1.0))
def test_invalid_completed_day_poisons_each_containing_month_then_rolls_out(
    field,
    value,
):
    panel = _panel(rows=700, stocks=2)
    starts = _eligible_month_starts(panel)
    poisoned_day = int(starts[1] + 5)
    panel[field][poisoned_day, 0] = value
    result = Completed52WeekHighProximityStrict().calc_batch(panel)

    poisoned_states = []
    for month_start in starts:
        month_start = int(month_start)
        month_end = _month_end(panel, month_start)
        should_poison = month_start - 252 <= poisoned_day < month_start
        poisoned_states.append(should_poison)
        if should_poison:
            assert np.isnan(result[month_start:month_end, 0]).all()
        else:
            assert np.isfinite(result[month_start:month_end, 0]).all()
        assert np.isfinite(result[month_start:month_end, 1]).all()

    first_poison = poisoned_states.index(True)
    last_poison = len(poisoned_states) - 1 - poisoned_states[::-1].index(True)
    assert not any(poisoned_states[:first_poison])
    assert not any(poisoned_states[last_poison + 1 :])
    assert last_poison + 1 < len(poisoned_states)


def test_current_open_and_st_are_legality_gates_only():
    panel = _panel(rows=420, stocks=5)
    factor = Completed52WeekHighProximityStrict()
    month_start = int(_eligible_month_starts(panel)[0])
    expected = factor.calc_batch(panel)[month_start]
    assert np.isfinite(expected).all()

    panel["open"][month_start] = np.array([2.0, 2000.0, 1.99, np.nan, np.inf])
    panel["st_mask"][month_start, 1] = True
    actual = factor.calc_batch(panel)[month_start]

    np.testing.assert_allclose(actual[0], expected[0])
    assert np.isnan(actual[1:]).all()


def test_does_not_shrink_the_window_before_a_month_has_252_completed_rows():
    panel = _panel(rows=252, stocks=3)
    result = Completed52WeekHighProximityStrict().calc_batch(panel)
    assert np.isnan(result).all()


@pytest.mark.parametrize("field", ("open", "close", "preClose", "st_mask"))
def test_rejects_panel_shape_mismatch(field):
    panel = _panel()
    panel[field] = panel[field][:-1]
    with pytest.raises(ValueError, match="matching shapes"):
        Completed52WeekHighProximityStrict().calc_batch(panel)


@pytest.mark.parametrize("field", ("open", "close", "preClose"))
def test_rejects_non_matrix_price_panel(field):
    panel = _panel()
    panel[field] = panel[field][:, 0]
    with pytest.raises(ValueError, match="two-dimensional"):
        Completed52WeekHighProximityStrict().calc_batch(panel)


@pytest.mark.parametrize("field", ("open", "close", "preClose"))
def test_rejects_non_real_numeric_price_dtype(field):
    panel = _panel()
    panel[field] = panel[field].astype(str)
    with pytest.raises(ValueError, match="real numeric dtype"):
        Completed52WeekHighProximityStrict().calc_batch(panel)


def test_rejects_non_boolean_st_mask():
    panel = _panel()
    panel["st_mask"] = panel["st_mask"].astype(np.int8)
    with pytest.raises(ValueError, match="boolean"):
        Completed52WeekHighProximityStrict().calc_batch(panel)


def test_accepts_production_style_python_date_and_datetime_lists():
    panel = _panel()
    factor = Completed52WeekHighProximityStrict()
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


def test_rejects_trade_date_shape_length_dtype_invalid_string_and_nat():
    factor = Completed52WeekHighProximityStrict()

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
        Completed52WeekHighProximityStrict().calc_batch(panel)


def test_rejects_missing_calendar_month_in_trade_dates():
    panel = _panel(rows=600)
    omitted_month = np.datetime64("2019-06", "M")
    keep = panel["trade_dates"].astype("datetime64[M]") != omitted_month
    original_rows = len(keep)
    for name, values in list(panel.items()):
        if values.ndim >= 1 and len(values) == original_rows:
            panel[name] = values[keep]

    with pytest.raises(ValueError, match="consecutive calendar months"):
        Completed52WeekHighProximityStrict().calc_batch(panel)


@pytest.mark.parametrize(("rows", "stocks"), ((0, 3), (20, 0)))
def test_rejects_empty_panel_dimensions(rows, stocks):
    panel = _panel(rows=rows, stocks=stocks)
    with pytest.raises(ValueError, match="at least one row and stock"):
        Completed52WeekHighProximityStrict().calc_batch(panel)


def test_420_by_1100_panel_is_vectorized_across_stocks():
    rows, stocks = 420, 1100
    rng = np.random.default_rng(20260722)
    trade_dates = _business_dates(rows)
    pre_close = rng.uniform(2.0, 100.0, size=(rows, stocks))
    gross = rng.lognormal(0.0002, 0.015, size=(rows, stocks))
    panel = {
        "trade_dates": trade_dates,
        "open": rng.uniform(2.0, 100.0, size=(rows, stocks)),
        "close": pre_close * gross,
        "preClose": pre_close,
        "st_mask": np.zeros((rows, stocks), dtype=bool),
    }

    started = time.perf_counter()
    result = Completed52WeekHighProximityStrict().calc_batch(panel)
    elapsed = time.perf_counter() - started

    first_usable = int(_eligible_month_starts(panel)[0])
    assert result.shape == (rows, stocks)
    assert result.dtype == np.float32
    assert np.isnan(result[:first_usable]).all()
    assert np.isfinite(result[first_usable:]).all()
    assert np.all((result[first_usable:] > 0.0) & (result[first_usable:] <= 1.0))
    assert elapsed < 1.0
