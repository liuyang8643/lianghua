import time
import tracemalloc

import numpy as np
import pytest

from factor_db.factors.AmihudIlliquidityStrict import AmihudIlliquidityStrict
from factor_db.factors.CompletedSignedAmountImbalance20Strict import (
    CompletedSignedAmountImbalance20Strict,
)


def _panel(rows=45, stocks=4):
    day = np.arange(rows, dtype=np.float64)[:, None]
    stock = np.arange(stocks, dtype=np.float64)[None, :]
    pre_close = 8.0 + 0.01 * day + stock
    direction = ((day + stock) % 3.0) - 1.0
    close = pre_close * (1.0 + 0.01 * direction)
    amount = (1.0e8 + 1.0e6 * day) * (1.0 + 0.05 * stock)
    open_ = pre_close.copy()
    return {
        "open": open_,
        "high": np.maximum(open_, close) * 1.01,
        "low": np.minimum(open_, close) * 0.99,
        "close": close,
        "preClose": pre_close,
        "volume": amount / open_,
        "amount": amount,
    }


def _brute_force(panel):
    close = np.asarray(panel["close"], dtype=np.float64)
    pre_close = np.asarray(panel["preClose"], dtype=np.float64)
    amount = np.asarray(panel["amount"], dtype=np.float64)
    expected = np.full(close.shape, np.nan, dtype=np.float64)
    for row in range(20, close.shape[0]):
        for stock in range(close.shape[1]):
            close_window = close[row - 20 : row, stock]
            pre_close_window = pre_close[row - 20 : row, stock]
            amount_window = amount[row - 20 : row, stock]
            valid = (
                np.isfinite(close_window)
                & (close_window > 0.0)
                & np.isfinite(pre_close_window)
                & (pre_close_window > 0.0)
                & np.isfinite(amount_window)
                & (amount_window > 0.0)
            )
            if not np.all(valid):
                continue
            signs = np.where(
                close_window > pre_close_window,
                1.0,
                np.where(close_window < pre_close_window, -1.0, 0.0),
            )
            scale = np.max(amount_window)
            normalized_amount = amount_window / scale
            expected[row, stock] = np.sum(
                normalized_amount * signs, dtype=np.float64
            ) / np.sum(normalized_amount, dtype=np.float64)
    return expected


def test_metadata_and_exact_hand_oracle_include_flat_day_amount():
    factor = CompletedSignedAmountImbalance20Strict()
    assert factor.hist_days == 20
    assert factor.pre_ranked is False
    assert factor.requires_full_history is False

    pre_close = np.full((21, 1), 10.0, dtype=np.float64)
    close = pre_close.copy()
    amount = np.arange(1.0, 22.0, dtype=np.float64)[:, None]
    close[0, 0] = 11.0
    close[1, 0] = 9.0
    # Day 2 is flat and its amount of 3 remains in the denominator.
    result = factor.calc_batch(
        {"close": close, "preClose": pre_close, "amount": amount}
    )
    expected = np.float32((1.0 - 2.0) / np.sum(amount[:20, 0]))

    assert result.shape == close.shape
    assert result.dtype == np.float32
    assert np.isnan(result[:20]).all()
    assert result[20, 0] == expected


def test_first_valid_row_and_window_boundaries_are_exact():
    panel = _panel(rows=22, stocks=1)
    panel["close"][:] = panel["preClose"]
    factor = CompletedSignedAmountImbalance20Strict()
    baseline = factor.calc_batch(panel)[:, 0]

    oldest_changed = {name: values.copy() for name, values in panel.items()}
    oldest_changed["close"][0, 0] *= 1.1
    oldest_result = factor.calc_batch(oldest_changed)[:, 0]
    assert np.isnan(oldest_result[:20]).all()
    assert oldest_result[20] > baseline[20]
    assert oldest_result[21] == baseline[21]

    current_changed = {name: values.copy() for name, values in panel.items()}
    current_changed["close"][20, 0] *= 0.9
    current_result = factor.calc_batch(current_changed)[:, 0]
    assert current_result[20] == baseline[20]
    assert current_result[21] < baseline[21]


def test_t_row_and_future_perturbations_cannot_change_current_or_prefix_scores():
    panel = _panel(rows=70, stocks=4)
    factor = CompletedSignedAmountImbalance20Strict()
    expected = factor.calc_batch(panel)
    changed = {name: values.copy() for name, values in panel.items()}
    row = 35
    for name in ("open", "high", "low", "close", "volume", "amount"):
        changed[name][row:] = np.nan
    changed["preClose"][row:] = -np.inf
    actual = factor.calc_batch(changed)

    np.testing.assert_array_equal(actual[: row + 1], expected[: row + 1])


def test_truncating_future_rows_preserves_the_entire_prefix_exactly():
    panel = _panel(rows=620, stocks=7)
    factor = CompletedSignedAmountImbalance20Strict()
    full = factor.calc_batch(panel)
    prefix_rows = 413
    truncated = {
        name: values[:prefix_rows].copy() for name, values in panel.items()
    }
    prefix = factor.calc_batch(truncated)

    np.testing.assert_array_equal(prefix, full[:prefix_rows])


@pytest.mark.parametrize("field", ("close", "preClose", "amount"))
@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf, 0.0, -1.0))
def test_each_invalid_input_kind_poisons_every_affected_full_window(field, value):
    panel = _panel(rows=43, stocks=1)
    panel[field][7, 0] = value
    result = CompletedSignedAmountImbalance20Strict().calc_batch(panel)[:, 0]

    assert np.isnan(result[20:28]).all()
    assert np.isfinite(result[28])


@pytest.mark.parametrize("field", ("close", "preClose", "amount"))
def test_shape_mismatch_fails_loudly(field):
    panel = _panel()
    panel[field] = panel[field][:-1]
    with pytest.raises(ValueError, match="matching shapes"):
        CompletedSignedAmountImbalance20Strict().calc_batch(panel)


@pytest.mark.parametrize("field", ("close", "preClose", "amount"))
def test_non_matrix_input_fails_loudly(field):
    panel = _panel()
    panel[field] = panel[field][:, 0]
    message = "two-dimensional" if field == "close" else "matching shapes"
    with pytest.raises(ValueError, match=message):
        CompletedSignedAmountImbalance20Strict().calc_batch(panel)


def test_output_dtype_shape_and_mathematical_bounds():
    rows = 80
    pre_close = np.full((rows, 3), 10.0, dtype=np.float64)
    close = pre_close.copy()
    close[:, 0] = 11.0
    close[:, 1] = 9.0
    amount = np.linspace(1.0, 1.0e8, rows)[:, None] * np.ones((1, 3))
    result = CompletedSignedAmountImbalance20Strict().calc_batch(
        {"close": close, "preClose": pre_close, "amount": amount}
    )

    assert result.shape == close.shape
    assert result.dtype == np.float32
    assert np.isnan(result[:20]).all()
    assert np.isfinite(result[20:]).all()
    assert np.all((result[20:] >= -1.0) & (result[20:] <= 1.0))
    np.testing.assert_array_equal(result[20:, 0], np.float32(1.0))
    np.testing.assert_array_equal(result[20:, 1], np.float32(-1.0))
    np.testing.assert_array_equal(result[20:, 2], np.float32(0.0))


def test_direction_comparison_avoids_price_ratio_overflow_and_underflow():
    rows = 21
    tiny = np.nextafter(np.float64(0.0), np.float64(1.0))
    largest = np.finfo(np.float64).max
    close = np.column_stack(
        (np.full(rows, largest), np.full(rows, tiny))
    )
    pre_close = np.column_stack(
        (np.full(rows, tiny), np.full(rows, largest))
    )
    amount = np.ones((rows, 2), dtype=np.float64)

    result = CompletedSignedAmountImbalance20Strict().calc_batch(
        {"close": close, "preClose": pre_close, "amount": amount}
    )

    np.testing.assert_array_equal(result[20], np.array([1.0, -1.0], np.float32))


@pytest.mark.parametrize("dominant_amount", (1.0e16, 1.0e17))
def test_dominant_amount_leaving_window_does_not_lose_small_observations(
    dominant_amount,
):
    rows = 22
    pre_close = np.full((rows, 1), 10.0, dtype=np.float64)
    close = pre_close.copy()
    close[0, 0] = 11.0
    close[1, 0] = 9.0
    amount = np.ones((rows, 1), dtype=np.float64)
    amount[0, 0] = dominant_amount

    result = CompletedSignedAmountImbalance20Strict().calc_batch(
        {"close": close, "preClose": pre_close, "amount": amount}
    )

    assert np.isfinite(result[20:, 0]).all()
    assert result[21, 0] == np.float32(-1.0 / 20.0)


def test_every_finite_positive_amount_scale_produces_finite_scores():
    rows, stocks = 65, 3
    tiny = np.nextafter(np.float64(0.0), np.float64(1.0))
    largest = np.finfo(np.float64).max
    pre_close = np.full((rows, stocks), 10.0, dtype=np.float64)
    direction = (np.arange(rows)[:, None] + np.arange(stocks)[None, :]) % 3 - 1
    close = pre_close * (1.0 + 0.01 * direction)
    amount = np.empty((rows, stocks), dtype=np.float64)
    amount[:, 0] = largest
    amount[:, 1] = tiny
    amount[:, 2] = np.where(np.arange(rows) % 21 == 0, largest, tiny)
    panel = {"close": close, "preClose": pre_close, "amount": amount}

    actual = CompletedSignedAmountImbalance20Strict().calc_batch(panel)
    expected = _brute_force(panel)

    assert np.isfinite(actual[20:]).all()
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-6,
        atol=0.0,
        equal_nan=True,
    )


def test_randomized_panel_matches_independent_brute_force_oracle():
    rng = np.random.default_rng(20260722)
    rows, stocks = 620, 13
    pre_close = rng.uniform(1.0, 200.0, size=(rows, stocks))
    direction = rng.integers(-1, 2, size=(rows, stocks))
    close = pre_close * (1.0 + 0.03 * direction)
    amount = rng.lognormal(18.0, 1.2, size=(rows, stocks))
    panel = {"close": close, "preClose": pre_close, "amount": amount}
    invalid_rows = rng.choice(rows * stocks, size=90, replace=False)
    invalid_fields = rng.integers(0, 3, size=invalid_rows.size)
    invalid_values = (np.nan, np.inf, 0.0, -1.0)
    for index, field_index in zip(invalid_rows, invalid_fields):
        row, stock = divmod(int(index), stocks)
        field = ("close", "preClose", "amount")[int(field_index)]
        panel[field][row, stock] = invalid_values[(row + stock) % 4]

    actual = CompletedSignedAmountImbalance20Strict().calc_batch(panel)
    expected = _brute_force(panel)

    np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-6,
        atol=1e-7,
        equal_nan=True,
    )


def test_finite_mask_exactly_matches_amihud_on_randomized_ordinary_inputs():
    rng = np.random.default_rng(360020)
    rows, stocks = 900, 37
    pre_close = rng.uniform(1.0, 120.0, size=(rows, stocks))
    close = pre_close * rng.lognormal(0.0, 0.03, size=(rows, stocks))
    amount = rng.lognormal(18.0, 1.0, size=(rows, stocks))
    panel = {"close": close, "preClose": pre_close, "amount": amount}
    for field in ("close", "preClose", "amount"):
        flat = panel[field].reshape(-1)
        indices = rng.choice(flat.size, size=80, replace=False)
        flat[indices[:20]] = np.nan
        flat[indices[20:40]] = np.inf
        flat[indices[40:60]] = 0.0
        flat[indices[60:]] = -1.0

    actual = CompletedSignedAmountImbalance20Strict().calc_batch(panel)
    reference = AmihudIlliquidityStrict().calc_batch(panel)

    np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(reference))


def test_auxiliary_memory_is_bounded_by_window_and_stock_count():
    rows, stocks = 4000, 300
    close = np.full((rows, stocks), 10.0, dtype=np.float64)
    pre_close = np.full((rows, stocks), 10.0, dtype=np.float64)
    amount = np.full((rows, stocks), 1.0e8, dtype=np.float64)
    panel = {"close": close, "preClose": pre_close, "amount": amount}

    tracemalloc.start()
    result = CompletedSignedAmountImbalance20Strict().calc_batch(panel)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    auxiliary_peak = peak - result.nbytes

    assert result.shape == (rows, stocks)
    assert auxiliary_peak < 2_000_000


def test_5217_by_5000_panel_completes_under_one_second_when_memory_allows():
    rows, stocks = 5217, 5000
    try:
        close = np.ones((rows, stocks), dtype=np.float64)
        pre_close = np.ones((rows, stocks), dtype=np.float64)
        amount = (
            1.0e6
            + 17.0 * np.arange(rows, dtype=np.float64)[:, None]
            + np.arange(stocks, dtype=np.float64)[None, :]
        )
    except MemoryError:
        pytest.skip("production-size benchmark allocation is unavailable")
    panel = {"close": close, "preClose": pre_close, "amount": amount}

    started = time.perf_counter()
    result = CompletedSignedAmountImbalance20Strict().calc_batch(panel)
    elapsed = time.perf_counter() - started

    assert result.shape == close.shape
    assert result.dtype == np.float32
    assert np.isnan(result[:20]).all()
    np.testing.assert_array_equal(result[20:], np.float32(0.0))
    assert elapsed < 1.0
