import inspect
import time
import tracemalloc

import numpy as np
import pytest

from factor_db.factors.AmihudIlliquidityStrict import AmihudIlliquidityStrict
from factor_db.factors.CompletedAmountConditionedReversal20Strict import (
    CompletedAmountConditionedReversal20Strict,
)


def _panel(rows=65, stocks=5):
    day = np.arange(rows, dtype=np.float64)[:, None]
    stock = np.arange(stocks, dtype=np.float64)[None, :]
    pre_close = 8.0 + 0.01 * day + stock
    direction = ((day + stock) % 3.0) - 1.0
    close = pre_close * (1.0 + 0.01 * direction)
    amount = (1.0e8 + 1.0e6 * day) * (1.0 + 0.05 * stock)
    return {"close": close, "preClose": pre_close, "amount": amount}


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
            signs = np.greater(close_window, pre_close_window).astype(np.int8)
            signs -= np.less(close_window, pre_close_window)
            scale = np.max(amount_window)
            normalized_amount = amount_window / scale
            unweighted = np.sum(signs, dtype=np.int64) / 20.0
            weighted = np.sum(
                normalized_amount * signs, dtype=np.float64
            ) / np.sum(normalized_amount, dtype=np.float64)
            expected[row, stock] = unweighted - weighted
    return expected


def test_metadata_exact_hand_oracle_and_flat_amount_denominator():
    factor = CompletedAmountConditionedReversal20Strict()
    assert factor.hist_days == 20
    assert factor.pre_ranked is False
    assert factor.requires_full_history is False

    pre_close = np.full((21, 1), 10.0, dtype=np.float64)
    close = pre_close.copy()
    amount = np.ones((21, 1), dtype=np.float64)
    close[0, 0] = 11.0
    close[1, 0] = 9.0
    amount[1, 0] = 3.0
    # Day 2 is flat: its amount makes the full denominator exactly 32.
    amount[2, 0] = 11.0

    result = factor.calc_batch(
        {"close": close, "preClose": pre_close, "amount": amount}
    )

    assert result.shape == close.shape
    assert result.dtype == np.float32
    assert np.isnan(result[:20]).all()
    # U=(+1-1)/20=0, W=(1-3)/32=-1/16, score=U-W=+1/16.
    assert result[20, 0] == np.float32(1.0 / 16.0)


@pytest.mark.parametrize(
    "amount_scale",
    (
        np.nextafter(np.float64(0.0), np.float64(1.0)),
        np.float64(1.0),
        np.finfo(np.float64).max,
    ),
)
def test_equal_amount_identity_is_strictly_zero_at_every_positive_scale(
    amount_scale,
):
    rows, stocks = 83, 4
    pre_close = np.full((rows, stocks), 10.0, dtype=np.float64)
    directions = (
        np.arange(rows)[:, None] + 2 * np.arange(stocks)[None, :]
    ) % 3 - 1
    close = pre_close.copy()
    close[directions > 0] = 11.0
    close[directions < 0] = 9.0
    amount = np.full((rows, stocks), amount_scale, dtype=np.float64)

    result = CompletedAmountConditionedReversal20Strict().calc_batch(
        {"close": close, "preClose": pre_close, "amount": amount}
    )

    assert np.isnan(result[:20]).all()
    np.testing.assert_array_equal(result[20:], np.float32(0.0))


def test_flat_day_amount_changes_weighted_denominator_not_unweighted_signs():
    pre_close = np.full((21, 1), 10.0, dtype=np.float64)
    close = pre_close.copy()
    close[0, 0] = 11.0
    close[1, 0] = 9.0
    amount = np.ones((21, 1), dtype=np.float64)
    amount[1, 0] = 3.0
    baseline = CompletedAmountConditionedReversal20Strict().calc_batch(
        {"close": close, "preClose": pre_close, "amount": amount}
    )[20, 0]
    amount[2, 0] = 1.0e12
    with_heavy_flat = CompletedAmountConditionedReversal20Strict().calc_batch(
        {"close": close, "preClose": pre_close, "amount": amount}
    )[20, 0]

    assert baseline > 0.0
    assert with_heavy_flat > 0.0
    assert with_heavy_flat < baseline


def test_first_valid_row_and_sliding_window_boundaries_are_exact():
    panel = _panel(rows=23, stocks=1)
    factor = CompletedAmountConditionedReversal20Strict()
    baseline = factor.calc_batch(panel)[:, 0]
    expected = _brute_force(panel)[:, 0]
    np.testing.assert_allclose(baseline, expected, rtol=1e-6, atol=1e-7, equal_nan=True)
    assert np.isnan(baseline[:20]).all()
    assert np.isfinite(baseline[20:]).all()

    changed = {name: values.copy() for name, values in panel.items()}
    changed["amount"][0, 0] *= 1.0e6
    shifted = factor.calc_batch(changed)[:, 0]
    assert shifted[20] != baseline[20]
    np.testing.assert_array_equal(shifted[21:], baseline[21:])


def test_t_row_and_future_perturbations_cannot_change_current_or_prefix_scores():
    panel = _panel(rows=90, stocks=7)
    factor = CompletedAmountConditionedReversal20Strict()
    expected = factor.calc_batch(panel)
    row = 47
    changed = {name: values.copy() for name, values in panel.items()}
    changed["close"][row:] = np.nan
    changed["preClose"][row:] = -np.inf
    changed["amount"][row:] = 0.0
    actual = factor.calc_batch(changed)

    np.testing.assert_array_equal(actual[: row + 1], expected[: row + 1])


def test_truncating_future_rows_preserves_entire_prefix_bit_for_bit():
    panel = _panel(rows=641, stocks=11)
    factor = CompletedAmountConditionedReversal20Strict()
    full = factor.calc_batch(panel)
    prefix_rows = 417
    prefix = factor.calc_batch(
        {name: values[:prefix_rows].copy() for name, values in panel.items()}
    )

    np.testing.assert_array_equal(prefix, full[:prefix_rows])


@pytest.mark.parametrize("missing", ("close", "preClose", "amount"))
def test_missing_required_field_fails_loudly(missing):
    panel = _panel()
    del panel[missing]
    with pytest.raises(KeyError, match=missing):
        CompletedAmountConditionedReversal20Strict().calc_batch(panel)


@pytest.mark.parametrize("field", ("close", "preClose", "amount"))
def test_shape_mismatch_fails_loudly(field):
    panel = _panel()
    panel[field] = panel[field][:-1]
    with pytest.raises(ValueError, match="matching shapes"):
        CompletedAmountConditionedReversal20Strict().calc_batch(panel)


@pytest.mark.parametrize("field", ("close", "preClose", "amount"))
def test_non_matrix_input_fails_loudly(field):
    panel = _panel()
    panel[field] = panel[field][:, 0]
    message = "two-dimensional" if field == "close" else "matching shapes"
    with pytest.raises(ValueError, match=message):
        CompletedAmountConditionedReversal20Strict().calc_batch(panel)


@pytest.mark.parametrize("field", ("close", "preClose", "amount"))
@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf, 0.0, -1.0))
def test_each_invalid_value_poisons_every_affected_full_window(field, value):
    panel = _panel(rows=43, stocks=1)
    panel[field][7, 0] = value
    result = CompletedAmountConditionedReversal20Strict().calc_batch(panel)[:, 0]

    assert np.isnan(result[20:28]).all()
    assert np.isfinite(result[28])


def test_randomized_panels_match_independent_brute_force_oracle():
    for seed in range(12):
        rng = np.random.default_rng(370000 + seed)
        rows = int(rng.integers(41, 150))
        stocks = int(rng.integers(2, 15))
        pre_close = rng.uniform(1.0, 200.0, size=(rows, stocks))
        direction = rng.integers(-1, 2, size=(rows, stocks))
        close = pre_close.copy()
        close[direction > 0] *= 1.03
        close[direction < 0] *= 0.97
        exponents = rng.uniform(-200.0, 200.0, size=(rows, stocks))
        amount = np.power(10.0, exponents)
        panel = {"close": close, "preClose": pre_close, "amount": amount}
        invalid_locations = rng.choice(rows * stocks, size=min(20, rows), replace=False)
        for ordinal, location in enumerate(invalid_locations):
            row, stock = divmod(int(location), stocks)
            field = ("close", "preClose", "amount")[ordinal % 3]
            panel[field][row, stock] = (np.nan, np.inf, 0.0, -1.0)[ordinal % 4]

        actual = CompletedAmountConditionedReversal20Strict().calc_batch(panel)
        expected = _brute_force(panel)

        np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-6,
            atol=1e-7,
            equal_nan=True,
        )


def test_price_direction_comparison_handles_smallest_and_largest_float64():
    rows, stocks = 44, 2
    tiny = np.nextafter(np.float64(0.0), np.float64(1.0))
    largest = np.finfo(np.float64).max
    pre_close = np.empty((rows, stocks), dtype=np.float64)
    close = np.empty((rows, stocks), dtype=np.float64)
    pre_close[:, 0] = np.where(np.arange(rows) % 2 == 0, tiny, largest)
    close[:, 0] = np.where(np.arange(rows) % 2 == 0, largest, tiny)
    pre_close[:, 1] = close[:, 0]
    close[:, 1] = pre_close[:, 0]
    amount = 1.0 + np.arange(rows, dtype=np.float64)[:, None]
    amount = amount * np.ones((1, stocks), dtype=np.float64)
    panel = {"close": close, "preClose": pre_close, "amount": amount}

    actual = CompletedAmountConditionedReversal20Strict().calc_batch(panel)
    expected = _brute_force(panel)

    assert np.isfinite(actual[20:]).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7, equal_nan=True)


def test_smallest_largest_and_dominant_amounts_remain_finite_and_recover():
    rows, stocks = 42, 3
    tiny = np.nextafter(np.float64(0.0), np.float64(1.0))
    largest = np.finfo(np.float64).max
    pre_close = np.full((rows, stocks), 10.0, dtype=np.float64)
    direction = (np.arange(rows)[:, None] + np.arange(stocks)[None, :]) % 3 - 1
    close = pre_close.copy()
    close[direction > 0] = 11.0
    close[direction < 0] = 9.0
    amount = np.empty((rows, stocks), dtype=np.float64)
    amount[:, 0] = tiny
    amount[:, 1] = largest
    amount[:, 2] = tiny
    amount[0, 2] = largest
    panel = {"close": close, "preClose": pre_close, "amount": amount}

    actual = CompletedAmountConditionedReversal20Strict().calc_batch(panel)
    expected = _brute_force(panel)

    assert np.isfinite(actual[20:]).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7, equal_nan=True)

    # Once the dominant first row leaves, the remaining equal tiny amounts
    # recover the exact U == W identity instead of retaining subtraction loss.
    equal_after_exit = {
        "close": np.full((22, 1), 10.0, dtype=np.float64),
        "preClose": np.full((22, 1), 10.0, dtype=np.float64),
        "amount": np.ones((22, 1), dtype=np.float64),
    }
    equal_after_exit["close"][0, 0] = 11.0
    equal_after_exit["close"][1, 0] = 9.0
    equal_after_exit["amount"][0, 0] = largest
    recovered = CompletedAmountConditionedReversal20Strict().calc_batch(
        equal_after_exit
    )
    assert np.isfinite(recovered[20:, 0]).all()
    assert recovered[21, 0] == np.float32(0.0)


def test_mathematical_bound_is_enforced_by_formula_without_clipping():
    rows = 21
    pre_close = np.full((rows, 2), 10.0, dtype=np.float64)
    close = np.full((rows, 2), 11.0, dtype=np.float64)
    amount = np.ones((rows, 2), dtype=np.float64)
    close[0, 0] = 9.0
    amount[0, 0] = np.finfo(np.float64).max
    close[0, 1] = 11.0
    close[1:, 1] = 9.0
    amount[0, 1] = np.finfo(np.float64).max

    result = CompletedAmountConditionedReversal20Strict().calc_batch(
        {"close": close, "preClose": pre_close, "amount": amount}
    )

    assert np.isfinite(result[20]).all()
    assert np.all(result[20] >= -2.0)
    assert np.all(result[20] <= 2.0)
    np.testing.assert_allclose(result[20], np.asarray([1.9, -1.9], np.float32))
    source = inspect.getsource(CompletedAmountConditionedReversal20Strict.calc_batch)
    assert "clip" not in source


def test_factor_reads_only_the_three_preregistered_fields():
    class RecordingPanel(dict):
        def __init__(self, values):
            super().__init__(values)
            self.accessed = []

        def __getitem__(self, key):
            self.accessed.append(key)
            return super().__getitem__(key)

    panel = RecordingPanel(_panel())
    CompletedAmountConditionedReversal20Strict().calc_batch(panel)

    assert panel.accessed == ["close", "preClose", "amount"]


def test_finite_mask_exactly_matches_amihud_contract_on_randomized_inputs():
    rng = np.random.default_rng(372020)
    rows, stocks = 900, 37
    pre_close = rng.uniform(1.0, 120.0, size=(rows, stocks))
    close = pre_close * rng.lognormal(0.0, 0.03, size=(rows, stocks))
    amount = rng.lognormal(18.0, 1.0, size=(rows, stocks))
    panel = {"close": close, "preClose": pre_close, "amount": amount}
    for field in ("close", "preClose", "amount"):
        flat = panel[field].reshape(-1)
        indices = rng.choice(flat.size, size=100, replace=False)
        for block, value in enumerate((np.nan, np.inf, -np.inf, 0.0, -1.0)):
            flat[indices[block * 20 : (block + 1) * 20]] = value

    actual = CompletedAmountConditionedReversal20Strict().calc_batch(panel)
    reference = AmihudIlliquidityStrict().calc_batch(panel)

    np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(reference))


def _auxiliary_peak(rows, stocks=180):
    close = np.full((rows, stocks), 10.0, dtype=np.float64)
    pre_close = np.full((rows, stocks), 10.0, dtype=np.float64)
    amount = np.full((rows, stocks), 1.0e8, dtype=np.float64)
    tracemalloc.start()
    result = CompletedAmountConditionedReversal20Strict().calc_batch(
        {"close": close, "preClose": pre_close, "amount": amount}
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak - result.nbytes


def test_auxiliary_memory_is_window_bounded_and_does_not_grow_with_rows():
    peaks = [_auxiliary_peak(rows) for rows in (1000, 4000, 16000)]

    assert max(peaks) < 2_000_000
    assert max(peaks) - min(peaks) < 250_000


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

    started = time.perf_counter()
    result = CompletedAmountConditionedReversal20Strict().calc_batch(
        {"close": close, "preClose": pre_close, "amount": amount}
    )
    elapsed = time.perf_counter() - started

    assert result.shape == close.shape
    assert result.dtype == np.float32
    assert np.isnan(result[:20]).all()
    np.testing.assert_array_equal(result[20:], np.float32(0.0))
    assert elapsed < 1.0
