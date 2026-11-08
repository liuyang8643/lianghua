import time

import numpy as np
import pytest

from factor_db.factors.LongHorizonVol36MStrict import (
    LongHorizonVol36MStrict,
)


def _panel(months=43, stocks=4):
    first_month = np.datetime64("2017-01", "M")
    dates = []
    month_numbers = []
    within_month_days = []
    for month_number in range(months):
        month_start = (first_month + month_number).astype("datetime64[D]")
        dates.extend((month_start + 4, month_start + 14))
        month_numbers.extend((month_number, month_number))
        within_month_days.extend((0, 1))

    trade_dates = np.asarray(dates, dtype="datetime64[D]")
    month_number = np.asarray(month_numbers, dtype=np.float64)[:, None]
    within_month_day = np.asarray(within_month_days, dtype=np.float64)[:, None]
    stock = np.arange(stocks, dtype=np.float64)[None, :]
    pre_close = 8.0 + 0.01 * month_number + stock
    official_return = (
        0.0025 * np.sin(month_number / 3.0)
        + 0.0001 * stock
        + 0.00005 * within_month_day
    )
    close = pre_close * (1.0 + official_return)
    return {
        "trade_dates": trade_dates,
        "open": np.full(close.shape, 5.0, dtype=np.float64),
        "close": close,
        "preClose": pre_close,
        "st_mask": np.zeros(close.shape, dtype=bool),
        "high": close * 1.02,
        "low": close * 0.98,
        "volume": np.full(close.shape, 1e6, dtype=np.float64),
        "amount": np.full(close.shape, 1e8, dtype=np.float64),
    }


def _month_rows(panel, month_number):
    target = np.datetime64("2017-01", "M") + month_number
    return np.flatnonzero(
        panel["trade_dates"].astype("datetime64[M]") == target
    )


def _manual_score(panel, target_month, stock):
    dates = panel["trade_dates"].astype("datetime64[M]")
    target = np.datetime64("2017-01", "M") + target_month
    monthly_returns = []
    for month in range(target_month - 36, target_month):
        selected = dates == (np.datetime64("2017-01", "M") + month)
        gross = panel["close"][selected, stock] / panel["preClose"][selected, stock]
        monthly_returns.append(np.expm1(np.sum(np.log(gross), dtype=np.float64)))
    assert len(monthly_returns) == 36
    assert dates[_month_rows(panel, target_month)[0]] == target
    return -np.std(monthly_returns, dtype=np.float64)


def test_uses_exactly_36_completed_consecutive_calendar_months():
    panel = _panel(months=39, stocks=3)
    factor = LongHorizonVol36MStrict()
    result = factor.calc_batch(panel)
    first_signal_rows = _month_rows(panel, 37)

    assert factor.hist_days >= 900
    assert result.shape == panel["close"].shape
    assert result.dtype == np.float32
    assert np.isnan(result[: first_signal_rows[0]]).all()
    for row in first_signal_rows:
        for stock in range(result.shape[1]):
            np.testing.assert_allclose(
                result[row, stock],
                _manual_score(panel, 37, stock),
                rtol=1e-6,
                atol=0.0,
            )


def test_first_panel_month_is_never_treated_as_complete_history():
    panel = _panel(months=38, stocks=1)
    factor = LongHorizonVol36MStrict()
    expected = factor.calc_batch(panel)
    changed = {name: values.copy() for name, values in panel.items()}
    changed["close"][_month_rows(panel, 0)] *= 1.8

    np.testing.assert_array_equal(factor.calc_batch(changed), expected)


def test_signal_is_fixed_within_month_and_changes_only_after_month_completes():
    panel = _panel(months=39, stocks=1)
    factor = LongHorizonVol36MStrict()
    baseline = factor.calc_batch(panel)
    target_rows = _month_rows(panel, 37)
    next_month_rows = _month_rows(panel, 38)

    changed = {name: values.copy() for name, values in panel.items()}
    changed["close"][target_rows[0], 0] *= 1.5
    actual = factor.calc_batch(changed)

    np.testing.assert_array_equal(actual[target_rows], baseline[target_rows])
    assert actual[next_month_rows[0], 0] != baseline[next_month_rows[0], 0]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("close", np.nan),
        ("close", np.inf),
        ("close", 0.0),
        ("preClose", np.nan),
        ("preClose", np.inf),
        ("preClose", -1.0),
    ),
)
def test_one_invalid_market_day_poisons_the_whole_month_and_36m_windows(
    field,
    value,
):
    panel = _panel(months=43, stocks=2)
    poisoned_month = 5
    panel[field][_month_rows(panel, poisoned_month)[0], 0] = value
    result = LongHorizonVol36MStrict().calc_batch(panel)

    for target_month in range(37, 42):
        rows = _month_rows(panel, target_month)
        assert np.isnan(result[rows, 0]).all()
        assert np.isfinite(result[rows, 1]).all()
    assert np.isfinite(result[_month_rows(panel, 42), 0]).all()


def test_equal_completed_day_price_scaling_is_corporate_action_invariant():
    panel = _panel(months=39, stocks=3)
    factor = LongHorizonVol36MStrict()
    target_rows = _month_rows(panel, 37)
    expected = factor.calc_batch(panel)[target_rows]

    scaled = {name: values.copy() for name, values in panel.items()}
    completed_day = _month_rows(panel, 20)[0]
    scaled["close"][completed_day] *= 0.2
    scaled["preClose"][completed_day] *= 0.2

    np.testing.assert_allclose(
        factor.calc_batch(scaled)[target_rows],
        expected,
        rtol=1e-6,
        atol=0.0,
    )


def test_current_day_hlcva_and_preclose_cannot_change_current_month_signal():
    panel = _panel(months=39, stocks=4)
    factor = LongHorizonVol36MStrict()
    target_rows = _month_rows(panel, 37)
    expected = factor.calc_batch(panel)[target_rows]
    changed = {name: values.copy() for name, values in panel.items()}
    current_row = target_rows[0]
    for name in ("high", "low", "close", "volume", "amount", "preClose"):
        changed[name][current_row] = np.array([np.nan, np.inf, -1.0, 1e30])

    np.testing.assert_array_equal(
        factor.calc_batch(changed)[target_rows],
        expected,
    )


def test_current_open_and_st_are_legality_gates_only():
    panel = _panel(months=39, stocks=4)
    factor = LongHorizonVol36MStrict()
    target_row = _month_rows(panel, 37)[0]
    expected = factor.calc_batch(panel)[target_row]
    assert np.isfinite(expected).all()

    panel["open"][target_row] = np.array([2.0, 2000.0, 1.99, np.nan])
    panel["st_mask"][target_row, 1] = True
    actual = factor.calc_batch(panel)[target_row]

    np.testing.assert_allclose(actual[0], expected[0])
    assert np.isnan(actual[1:]).all()


@pytest.mark.parametrize("field", ("open", "close", "preClose", "st_mask"))
def test_rejects_panel_shape_mismatch(field):
    panel = _panel()
    panel[field] = panel[field][:-1]
    with pytest.raises(ValueError, match="matching shapes"):
        LongHorizonVol36MStrict().calc_batch(panel)


@pytest.mark.parametrize("field", ("open", "close", "preClose"))
def test_rejects_non_matrix_price_panel(field):
    panel = _panel()
    panel[field] = panel[field][:, 0]
    with pytest.raises(ValueError, match="two-dimensional"):
        LongHorizonVol36MStrict().calc_batch(panel)


def test_rejects_non_boolean_st_mask():
    panel = _panel()
    panel["st_mask"] = panel["st_mask"].astype(np.int8)
    with pytest.raises(ValueError, match="boolean"):
        LongHorizonVol36MStrict().calc_batch(panel)


def test_rejects_trade_date_shape_length_and_invalid_values():
    factor = LongHorizonVol36MStrict()

    panel = _panel()
    panel["trade_dates"] = panel["trade_dates"][:-1]
    with pytest.raises(ValueError, match="length"):
        factor.calc_batch(panel)

    panel = _panel()
    panel["trade_dates"] = panel["trade_dates"][:, None]
    with pytest.raises(ValueError, match="one-dimensional"):
        factor.calc_batch(panel)

    panel = _panel()
    panel["trade_dates"] = panel["trade_dates"].copy()
    panel["trade_dates"][4] = np.datetime64("NaT")
    with pytest.raises(ValueError, match="NaT"):
        factor.calc_batch(panel)

    panel = _panel()
    panel["trade_dates"] = panel["trade_dates"].astype(object)
    panel["trade_dates"][3] = "not-a-date"
    with pytest.raises(ValueError, match="valid calendar dates"):
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
        LongHorizonVol36MStrict().calc_batch(panel)


def test_rejects_missing_calendar_month_in_trade_dates():
    panel = _panel()
    omitted_month = np.datetime64("2017-11", "M")
    keep = panel["trade_dates"].astype("datetime64[M]") != omitted_month
    for name, values in list(panel.items()):
        panel[name] = values[keep]

    with pytest.raises(ValueError, match="consecutive calendar months"):
        LongHorizonVol36MStrict().calc_batch(panel)


def test_medium_panel_is_vectorized_across_stocks():
    all_dates = np.arange(
        np.datetime64("2016-01-01"),
        np.datetime64("2024-01-01"),
        dtype="datetime64[D]",
    )
    trade_dates = all_dates[np.is_busday(all_dates)]
    rows, stocks = len(trade_dates), 1200
    day = np.arange(rows, dtype=np.float32)[:, None]
    stock = np.arange(stocks, dtype=np.float32)[None, :]
    pre_close = np.broadcast_to(
        8.0 + stock * 0.001,
        (rows, stocks),
    ).copy()
    official_return = (
        0.001 * np.sin(day / 37.0) + 0.00001 * (stock % 17.0)
    )
    panel = {
        "trade_dates": trade_dates,
        "open": pre_close.copy(),
        "close": pre_close * (1.0 + official_return),
        "preClose": pre_close,
        "st_mask": np.zeros((rows, stocks), dtype=bool),
    }

    started = time.perf_counter()
    result = LongHorizonVol36MStrict().calc_batch(panel)
    elapsed = time.perf_counter() - started

    assert result.shape == (rows, stocks)
    first_usable = np.flatnonzero(
        trade_dates.astype("datetime64[M]") == np.datetime64("2019-02", "M")
    )[0]
    assert np.isnan(result[:first_usable]).all()
    assert np.isfinite(result[first_usable:]).all()
    assert elapsed < 1.0
