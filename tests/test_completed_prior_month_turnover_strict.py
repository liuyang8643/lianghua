import time
from datetime import datetime

import numpy as np
import pytest

from factor_db.factors.CompletedPriorMonthTurnoverStrict import (
    CompletedPriorMonthTurnoverStrict,
)


_FIRST_MONTH = np.datetime64("2019-01", "M")


def _panel(months=7, stocks=4, rows_per_month=4):
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
    shape = (len(trade_dates), stocks)
    month = np.asarray(month_numbers, dtype=np.float64)[:, None]
    within_month = np.asarray(days_in_month, dtype=np.float64)[:, None]
    stock = np.arange(stocks, dtype=np.float64)[None, :]
    total_share = np.broadcast_to(100.0 + 5.0 * stock, shape).copy()
    daily_turnover = (
        0.01
        + 0.001 * month
        + 0.0001 * within_month
        + 0.00001 * stock
    )
    volume = daily_turnover * (100.0 * total_share)
    prices = np.broadcast_to(5.0 + 0.1 * stock, shape).copy()
    return {
        "trade_dates": trade_dates,
        "volume": volume,
        "total_share": total_share,
        "open": prices.copy(),
        "high": prices * 1.02,
        "low": prices * 0.98,
        "close": prices * 1.01,
        "amount": volume * prices * 100.0,
        "preClose": prices.copy(),
        "st_mask": np.zeros(shape, dtype=bool),
    }


def _copy_panel(panel):
    return {name: values.copy() for name, values in panel.items()}


def _month_rows(panel, month_number):
    target = _FIRST_MONTH + month_number
    return np.flatnonzero(
        np.asarray(panel["trade_dates"]).astype("datetime64[M]") == target
    )


def _manual_prior_month_score(panel, target_month, stock):
    prior_rows = _month_rows(panel, target_month - 1)
    turnover = (
        panel["volume"][prior_rows, stock]
        / (100.0 * panel["total_share"][prior_rows, stock])
    )
    return -np.mean(turnover, dtype=np.float64)


def test_prior_complete_month_matches_hand_calculation_and_output_contract():
    panel = _panel(months=4, stocks=2, rows_per_month=3)
    prior_rows = _month_rows(panel, 1)
    panel["volume"][prior_rows, 0] = np.array([200.0, 400.0, 800.0])
    panel["total_share"][prior_rows, 0] = np.array([100.0, 100.0, 200.0])
    panel["volume"][prior_rows, 1] = np.array([300.0, 600.0, 300.0])
    panel["total_share"][prior_rows, 1] = np.array([100.0, 200.0, 100.0])

    factor = CompletedPriorMonthTurnoverStrict()
    result = factor.calc_batch(panel)
    target_rows = _month_rows(panel, 2)
    expected = np.array(
        [
            -np.mean([0.02, 0.04, 0.04]),
            -np.mean([0.03, 0.03, 0.03]),
        ],
        dtype=np.float32,
    )

    assert factor.hist_days == 60
    assert factor.pre_ranked is False
    assert factor.requires_full_history is False
    assert result.shape == panel["volume"].shape
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
            _manual_prior_month_score(panel, 2, stock),
            rtol=1e-6,
            atol=0.0,
        )


def test_score_is_frozen_for_every_market_row_in_natural_month():
    panel = _panel(months=6, stocks=3, rows_per_month=7)
    result = CompletedPriorMonthTurnoverStrict().calc_batch(panel)

    for month_number in range(2, 6):
        rows = _month_rows(panel, month_number)
        assert len(rows) == 7
        np.testing.assert_array_equal(
            result[rows],
            np.broadcast_to(result[rows[0]], (len(rows), 3)),
        )
        assert np.isfinite(result[rows]).all()


def test_partial_first_boundary_month_is_never_a_valid_formation_month():
    full = _panel(months=4, stocks=3, rows_per_month=8)
    crop_start = _month_rows(full, 0)[3]
    panel = {
        name: values[crop_start:].copy()
        for name, values in full.items()
    }
    assert panel["trade_dates"][0] == np.datetime64("2019-01-04")

    result = CompletedPriorMonthTurnoverStrict().calc_batch(panel)
    first_month = _month_rows(panel, 0)
    second_month = _month_rows(panel, 1)
    third_month = _month_rows(panel, 2)

    assert np.isnan(result[first_month]).all()
    assert np.isnan(result[second_month]).all()
    assert np.isfinite(result[third_month]).all()
    for stock in range(3):
        np.testing.assert_allclose(
            result[third_month[0], stock],
            _manual_prior_month_score(panel, 2, stock),
            rtol=1e-6,
            atol=0.0,
        )


def test_midmonth_crop_with_complete_prior_month_preserves_target_score():
    full = _panel(months=7, stocks=4, rows_per_month=20)
    factor = CompletedPriorMonthTurnoverStrict()
    expected = factor.calc_batch(full)

    crop_start = _month_rows(full, 2)[9]
    cropped = {
        name: values[crop_start:].copy()
        for name, values in full.items()
    }
    assert cropped["trade_dates"][0].astype("datetime64[M]") == _FIRST_MONTH + 2
    assert cropped["trade_dates"][0].astype("datetime64[D]") > (
        (_FIRST_MONTH + 2).astype("datetime64[D]")
    )
    actual = factor.calc_batch(cropped)

    for target_month in (4, 5, 6):
        full_rows = _month_rows(full, target_month)
        cropped_rows = _month_rows(cropped, target_month)
        np.testing.assert_array_equal(actual[cropped_rows], expected[full_rows])


@pytest.mark.parametrize("field", ("volume", "total_share"))
@pytest.mark.parametrize("value", (np.nan, 0.0, -1.0, np.inf))
def test_invalid_row_poisons_whole_next_month_only(field, value):
    panel = _panel(months=7, stocks=2, rows_per_month=5)
    factor = CompletedPriorMonthTurnoverStrict()
    baseline = factor.calc_batch(panel)
    poisoned_month = 3
    poisoned_row = _month_rows(panel, poisoned_month)[2]
    panel[field][poisoned_row, 0] = value
    actual = factor.calc_batch(panel)

    source_rows = _month_rows(panel, poisoned_month)
    next_rows = _month_rows(panel, poisoned_month + 1)
    recovered_rows = _month_rows(panel, poisoned_month + 2)
    np.testing.assert_array_equal(actual[source_rows], baseline[source_rows])
    assert np.isnan(actual[next_rows, 0]).all()
    np.testing.assert_array_equal(actual[next_rows, 1], baseline[next_rows, 1])
    np.testing.assert_array_equal(actual[recovered_rows], baseline[recovered_rows])


@pytest.mark.parametrize(
    "field",
    (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "preClose",
        "total_share",
    ),
)
def test_t_day_and_future_market_mutations_cannot_change_frozen_signal(field):
    panel = _panel(months=7, stocks=4, rows_per_month=5)
    factor = CompletedPriorMonthTurnoverStrict()
    baseline = factor.calc_batch(panel)
    target_month = 4
    target_rows = _month_rows(panel, target_month)
    changed = _copy_panel(panel)
    changed[field][target_rows[0] :] = np.nan
    actual = factor.calc_batch(changed)

    np.testing.assert_array_equal(actual[: target_rows[-1] + 1], baseline[: target_rows[-1] + 1])


def test_t_day_and_future_st_flags_are_not_factor_inputs():
    panel = _panel(months=7, stocks=4, rows_per_month=5)
    factor = CompletedPriorMonthTurnoverStrict()
    baseline = factor.calc_batch(panel)
    target_rows = _month_rows(panel, 4)
    panel["st_mask"][target_rows[0] :] = True

    actual = factor.calc_batch(panel)
    np.testing.assert_array_equal(actual[: target_rows[-1] + 1], baseline[: target_rows[-1] + 1])


def test_volume_lots_to_total_share_unit_conversion_is_exactly_100():
    panel = _panel(months=3, stocks=1, rows_per_month=2)
    formation_rows = _month_rows(panel, 1)
    panel["volume"][formation_rows, 0] = np.array([100.0, 300.0])
    panel["total_share"][formation_rows, 0] = np.array([2.0, 2.0])

    result = CompletedPriorMonthTurnoverStrict().calc_batch(panel)
    target_rows = _month_rows(panel, 2)
    # Daily rates are 0.5 and 1.5, so their arithmetic mean is exactly 1.
    np.testing.assert_array_equal(result[target_rows, 0], np.float32(-1.0))


@pytest.mark.parametrize("field", ("volume", "total_share"))
def test_rejects_panel_shape_mismatch(field):
    panel = _panel()
    panel[field] = panel[field][:-1]
    with pytest.raises(ValueError, match="matching shapes"):
        CompletedPriorMonthTurnoverStrict().calc_batch(panel)


@pytest.mark.parametrize("field", ("volume", "total_share"))
def test_rejects_non_matrix_numeric_inputs(field):
    panel = _panel()
    panel[field] = panel[field][:, 0]
    with pytest.raises(ValueError, match="two-dimensional"):
        CompletedPriorMonthTurnoverStrict().calc_batch(panel)


@pytest.mark.parametrize("field", ("volume", "total_share"))
@pytest.mark.parametrize("bad_dtype", (str, bool, np.complex128))
def test_rejects_non_real_numeric_input_dtypes(field, bad_dtype):
    panel = _panel()
    panel[field] = panel[field].astype(bad_dtype)
    with pytest.raises(ValueError, match="real numeric dtype"):
        CompletedPriorMonthTurnoverStrict().calc_batch(panel)


def test_accepts_python_date_and_datetime_trade_dates():
    panel = _panel()
    factor = CompletedPriorMonthTurnoverStrict()
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


def test_rejects_bad_trade_date_shape_length_dtype_value_and_nat():
    factor = CompletedPriorMonthTurnoverStrict()

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
        CompletedPriorMonthTurnoverStrict().calc_batch(panel)


def test_rejects_missing_calendar_month_in_trade_dates():
    panel = _panel()
    omitted_month = _FIRST_MONTH + 3
    keep = panel["trade_dates"].astype("datetime64[M]") != omitted_month
    original_rows = len(keep)
    for name, values in list(panel.items()):
        if values.ndim >= 1 and len(values) == original_rows:
            panel[name] = values[keep]

    with pytest.raises(ValueError, match="consecutive calendar months"):
        CompletedPriorMonthTurnoverStrict().calc_batch(panel)


@pytest.mark.parametrize(("rows", "stocks"), ((0, 3), (4, 0)))
def test_rejects_empty_panel_dimensions(rows, stocks):
    panel = _panel(months=1, stocks=stocks, rows_per_month=rows)
    with pytest.raises(ValueError, match="at least one row and stock"):
        CompletedPriorMonthTurnoverStrict().calc_batch(panel)


def test_1200_by_5000_panel_is_vectorized_at_production_like_scale():
    months = 60
    rows_per_month = 20
    stocks = 5000
    dates = []
    for month_number in range(months):
        month_start = (_FIRST_MONTH + month_number).astype("datetime64[D]")
        dates.extend(month_start + np.arange(rows_per_month))
    trade_dates = np.asarray(dates, dtype="datetime64[D]")
    shape = (len(trade_dates), stocks)
    panel = {
        "trade_dates": trade_dates,
        "volume": np.full(shape, 1_000_000.0, dtype=np.float32),
        "total_share": np.full(shape, 100_000_000.0, dtype=np.float32),
    }

    started = time.perf_counter()
    result = CompletedPriorMonthTurnoverStrict().calc_batch(panel)
    elapsed = time.perf_counter() - started

    first_usable = 2 * rows_per_month
    assert result.shape == shape
    assert result.dtype == np.float32
    assert np.isnan(result[:first_usable]).all()
    assert np.isfinite(result[first_usable:]).all()
    np.testing.assert_array_equal(
        result[first_usable:],
        np.float32(-0.0001),
    )
    assert elapsed < 1.0
