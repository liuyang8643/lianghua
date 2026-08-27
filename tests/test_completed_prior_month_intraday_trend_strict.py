import ast
import inspect
import textwrap

import numpy as np
import pytest

from factor_db.factors.CompletedPriorMonthIntradayTrendStrict import (
    CompletedPriorMonthIntradayTrendStrict,
)


_FIRST_MONTH = np.datetime64("2023-01", "M")


def _panel(months=6, stocks=3, rows_per_month=4):
    dates = []
    month_numbers = []
    within_month_days = []
    for month_number in range(months):
        month_start = (_FIRST_MONTH + month_number).astype("datetime64[D]")
        for within_month_day in range(rows_per_month):
            dates.append(month_start + within_month_day)
            month_numbers.append(month_number)
            within_month_days.append(within_month_day)

    trade_dates = np.asarray(dates, dtype="datetime64[D]")
    shape = (len(trade_dates), stocks)
    month = np.asarray(month_numbers, dtype=np.float64)[:, None]
    within_month = np.asarray(within_month_days, dtype=np.float64)[:, None]
    stock = np.arange(stocks, dtype=np.float64)[None, :]
    open_price = np.broadcast_to(10.0 + 0.1 * stock + 0.01 * month, shape).copy()
    daily_return = (
        0.004
        + 0.0004 * month
        + 0.0002 * stock
        + 0.002 * np.sin((within_month + 1.0) * (stock + 1.0))
    )
    close = open_price * np.exp(daily_return)
    return {
        "trade_dates": trade_dates,
        "open": open_price,
        "close": close,
    }


def _copy_panel(panel):
    return {name: values.copy() for name, values in panel.items()}


def _month_rows(panel, month_number):
    target = _FIRST_MONTH + month_number
    return np.flatnonzero(
        np.asarray(panel["trade_dates"]).astype("datetime64[M]") == target
    )


def _manual_q(panel, formation_month, stock):
    rows = _month_rows(panel, formation_month)
    daily_return = np.log(
        panel["close"][rows, stock] / panel["open"][rows, stock]
    )
    return daily_return.sum() / np.sqrt(
        len(rows) * np.square(daily_return).sum()
    )


def test_matches_independent_prior_complete_month_q_and_output_contract():
    panel = _panel(months=4, stocks=2, rows_per_month=4)
    formation_rows = _month_rows(panel, 1)
    returns = np.asarray(
        [
            [0.020, -0.010],
            [-0.005, 0.015],
            [0.030, -0.002],
            [0.010, 0.012],
        ],
        dtype=np.float64,
    )
    panel["close"][formation_rows] = (
        panel["open"][formation_rows] * np.exp(returns)
    )

    factor = CompletedPriorMonthIntradayTrendStrict()
    result = factor.calc_batch(panel)
    target_rows = _month_rows(panel, 2)
    expected = np.asarray(
        [_manual_q(panel, 1, stock) for stock in range(2)],
        dtype=np.float32,
    )

    assert factor.hist_days == 60
    assert factor.update_frequency == "monthly"
    assert factor.pre_ranked is False
    assert factor.requires_full_history is False
    assert result.shape == panel["open"].shape
    assert result.dtype == np.float32
    np.testing.assert_allclose(
        result[target_rows],
        np.broadcast_to(expected, (len(target_rows), 2)),
        rtol=1e-6,
        atol=0.0,
    )


def test_first_panel_month_cannot_receive_or_supply_a_strict_score():
    full = _panel(months=4, stocks=3, rows_per_month=6)
    crop_start = _month_rows(full, 0)[2]
    panel = {
        name: values[crop_start:].copy()
        for name, values in full.items()
    }
    result = CompletedPriorMonthIntradayTrendStrict().calc_batch(panel)

    assert np.isnan(result[_month_rows(panel, 0)]).all()
    assert np.isnan(result[_month_rows(panel, 1)]).all()
    assert np.isfinite(result[_month_rows(panel, 2)]).all()


def test_score_is_frozen_throughout_each_target_natural_month():
    panel = _panel(months=7, stocks=4, rows_per_month=8)
    result = CompletedPriorMonthIntradayTrendStrict().calc_batch(panel)

    for month_number in range(2, 7):
        rows = _month_rows(panel, month_number)
        np.testing.assert_array_equal(
            result[rows],
            np.broadcast_to(result[rows[0]], (len(rows), 4)),
        )


def test_current_and_future_month_prices_cannot_change_current_month_score():
    panel = _panel(months=7, stocks=3, rows_per_month=4)
    factor = CompletedPriorMonthIntradayTrendStrict()
    baseline = factor.calc_batch(panel)
    target_month = 4
    target_rows = _month_rows(panel, target_month)
    next_rows = _month_rows(panel, target_month + 1)

    changed = _copy_panel(panel)
    changed["open"][target_rows] = 20.0
    replacement_returns = np.asarray(
        [0.10, -0.05, 0.08, -0.02],
        dtype=np.float64,
    )[:, None]
    changed["close"][target_rows] = 20.0 * np.exp(replacement_returns)
    changed_result = factor.calc_batch(changed)

    np.testing.assert_array_equal(
        changed_result[target_rows],
        baseline[target_rows],
    )
    assert not np.allclose(
        changed_result[next_rows],
        baseline[next_rows],
        equal_nan=True,
    )

    future_only = _copy_panel(panel)
    future_only["open"][next_rows[0] :] *= 7.0
    future_only["close"][next_rows[0] :] *= 0.3
    future_result = factor.calc_batch(future_only)
    np.testing.assert_array_equal(
        future_result[target_rows],
        baseline[target_rows],
    )


@pytest.mark.parametrize("field", ("open", "close"))
@pytest.mark.parametrize("value", (np.nan, np.inf, 0.0, -1.0))
def test_invalid_historical_price_poisons_whole_next_month_only(field, value):
    panel = _panel(months=7, stocks=2, rows_per_month=5)
    factor = CompletedPriorMonthIntradayTrendStrict()
    baseline = factor.calc_batch(panel)
    formation_month = 3
    formation_rows = _month_rows(panel, formation_month)
    target_rows = _month_rows(panel, formation_month + 1)
    recovered_rows = _month_rows(panel, formation_month + 2)
    panel[field][formation_rows[2], 0] = value

    actual = factor.calc_batch(panel)

    np.testing.assert_array_equal(
        actual[formation_rows],
        baseline[formation_rows],
    )
    assert np.isnan(actual[target_rows, 0]).all()
    np.testing.assert_array_equal(
        actual[target_rows, 1],
        baseline[target_rows, 1],
    )
    np.testing.assert_array_equal(
        actual[recovered_rows],
        baseline[recovered_rows],
    )


def test_zero_energy_and_nonpositive_q_remain_nan_in_next_month():
    panel = _panel(months=4, stocks=3, rows_per_month=4)
    formation_rows = _month_rows(panel, 1)
    panel["close"][formation_rows, 0] = panel["open"][formation_rows, 0]
    panel["close"][formation_rows, 1] = (
        panel["open"][formation_rows, 1] * np.exp(-0.01)
    )

    result = CompletedPriorMonthIntradayTrendStrict().calc_batch(panel)
    target_rows = _month_rows(panel, 2)

    assert np.isnan(result[target_rows, :2]).all()
    assert np.isfinite(result[target_rows, 2]).all()


class _RecordingPanel:
    def __init__(self, data):
        self._data = data
        self.accessed = []

    def __getitem__(self, key):
        self.accessed.append(key)
        if key not in self._data:
            raise AssertionError(f"unexpected panel field access: {key}")
        return self._data[key]


def test_reads_only_open_close_and_trade_dates():
    panel = _panel()
    recording = _RecordingPanel(panel)

    CompletedPriorMonthIntradayTrendStrict().calc_batch(recording)

    assert recording.accessed == ["open", "close", "trade_dates"]


def test_calc_batch_has_no_per_stock_or_per_row_python_loop():
    source = textwrap.dedent(
        inspect.getsource(CompletedPriorMonthIntradayTrendStrict.calc_batch)
    )
    tree = ast.parse(source)
    loop_nodes = (
        ast.For,
        ast.While,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
    )

    assert not any(isinstance(node, loop_nodes) for node in ast.walk(tree))
    assert "nan_to_num" not in source


@pytest.mark.parametrize("field", ("open", "close"))
def test_rejects_shape_mismatch_and_non_matrix_input(field):
    panel = _panel()
    panel[field] = panel[field][:-1]
    with pytest.raises(ValueError, match="matching shapes"):
        CompletedPriorMonthIntradayTrendStrict().calc_batch(panel)

    panel = _panel()
    panel[field] = panel[field][:, 0]
    with pytest.raises(ValueError, match="two-dimensional"):
        CompletedPriorMonthIntradayTrendStrict().calc_batch(panel)


@pytest.mark.parametrize("field", ("open", "close"))
@pytest.mark.parametrize("bad_dtype", (str, bool, np.complex128))
def test_rejects_non_real_numeric_prices(field, bad_dtype):
    panel = _panel()
    panel[field] = panel[field].astype(bad_dtype)

    with pytest.raises(ValueError, match="real numeric dtype"):
        CompletedPriorMonthIntradayTrendStrict().calc_batch(panel)


def test_rejects_missing_field_bad_dates_and_missing_calendar_month():
    factor = CompletedPriorMonthIntradayTrendStrict()

    for field in ("open", "close", "trade_dates"):
        panel = _panel()
        del panel[field]
        with pytest.raises(KeyError):
            factor.calc_batch(panel)

    panel = _panel()
    panel["trade_dates"][3] = panel["trade_dates"][2]
    with pytest.raises(ValueError, match="strictly increasing"):
        factor.calc_batch(panel)

    panel = _panel()
    keep = (
        panel["trade_dates"].astype("datetime64[M]")
        != _FIRST_MONTH + 3
    )
    panel = {name: values[keep] for name, values in panel.items()}
    with pytest.raises(ValueError, match="consecutive calendar months"):
        factor.calc_batch(panel)
