"""Independent scalar formulas and causal/missingness contracts."""

import numpy as np
import pytest

from factor.library import (
    CompletedAmountImbalance20,
    CompletedCloseLocation20,
    CompletedMomentum252Skip21,
    CompletedReversal20,
)


FACTORS = (
    CompletedReversal20, CompletedMomentum252Skip21,
    CompletedAmountImbalance20, CompletedCloseLocation20,
)


def _panel(rows=300, stocks=3):
    rng = np.random.default_rng(123)
    pre_close = rng.uniform(5, 20, (rows, stocks))
    close = pre_close * rng.uniform(0.96, 1.04, (rows, stocks))
    return {
        "preClose": pre_close, "close": close,
        "low": close * rng.uniform(0.96, 0.99, (rows, stocks)),
        "high": close * rng.uniform(1.01, 1.04, (rows, stocks)),
        "amount": rng.uniform(1e6, 1e8, (rows, stocks)),
    }


@pytest.mark.parametrize("factor", FACTORS)
def test_matches_independent_exact_completed_window(factor):
    panel = _panel()
    result = factor().calc_batch(panel)
    assert np.isnan(result[:factor.hist_days]).all()
    for row in (factor.hist_days, 280, 299):
        skip = 21 if factor is CompletedMomentum252Skip21 else 0
        window = slice(row - factor.hist_days, row - skip)
        close = panel["close"][window]
        pre_close = panel["preClose"][window]
        if factor is CompletedReversal20:
            expected = 1 - np.prod(close / pre_close, axis=0)
        elif factor is CompletedMomentum252Skip21:
            expected = np.prod(close / pre_close, axis=0) - 1
        elif factor is CompletedAmountImbalance20:
            amount = panel["amount"][window]
            expected = np.sum(np.sign(close - pre_close) * amount, axis=0) / np.sum(amount, axis=0)
        else:
            low, high = panel["low"][window], panel["high"][window]
            expected = np.mean((close - low) / (high - low), axis=0)
        np.testing.assert_allclose(result[row], expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("factor", FACTORS)
def test_current_and_future_prices_do_not_change_decision(factor):
    panel = _panel()
    original = factor().calc_batch(panel)
    for matrix in panel.values():
        matrix[270:] = np.nan
    changed = factor().calc_batch(panel)
    np.testing.assert_array_equal(original[:271], changed[:271])


@pytest.mark.parametrize("factor", tuple(f for f in FACTORS if f is not CompletedMomentum252Skip21))
def test_missing_completed_close_invalidates_exact_window_and_recovers(factor):
    panel = _panel(rows=560)
    missing_row = 270
    panel["close"][missing_row, 0] = np.nan
    result = factor().calc_batch(panel)
    skip = 21 if factor is CompletedMomentum252Skip21 else 0
    start = missing_row + skip + 1
    stop = missing_row + factor.hist_days + 1
    assert np.isnan(result[start:stop, 0]).all()
    assert np.isfinite(result[start:stop, 1:]).all()
    assert np.isfinite(result[start - 1, 0])
    assert np.isfinite(result[stop, 0])


def test_momentum_skips_only_interior_gaps_and_preserves_fixed_window():
    panel = _panel(rows=560)
    original = CompletedMomentum252Skip21().calc_batch(panel)
    panel["close"][270, 0] = np.nan
    result = CompletedMomentum252Skip21().calc_batch(panel)
    # Missing last/first window endpoint; internal gap between them is allowed.
    assert np.isnan(result[292, 0])
    assert np.isnan(result[522, 0])
    assert np.isfinite(result[293:522, 0]).all()
    assert result[291, 0] == original[291, 0]
    assert result[523, 0] == original[523, 0]
    observed = panel["close"][48:279, 0] / panel["preClose"][48:279, 0]
    np.testing.assert_allclose(result[300, 0], np.prod(observed[np.isfinite(observed)]) - 1,
                               rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("field", ("close", "preClose"))
@pytest.mark.parametrize("missing", (np.nan, np.inf, 0, -1))
def test_momentum_requires_185_observations_without_extrapolation(field, missing):
    panel = _panel()
    panel[field][1:47, 0] = missing  # 231 - 46 = 185 valid, endpoints retained.
    factor = CompletedMomentum252Skip21()
    kept = np.r_[0, 47:231]
    expected = np.prod(panel["close"][kept, 0] / panel["preClose"][kept, 0]) - 1
    np.testing.assert_allclose(factor.calc_batch(panel)[252, 0], expected, rtol=1e-10, atol=1e-12)
    panel[field][47, 0] = missing
    assert np.isnan(factor.calc_batch(panel)[252, 0])


@pytest.mark.parametrize("rows", (1, 100, 230, 231, 251, 252))
def test_momentum_short_history_remains_missing(rows):
    assert np.isnan(CompletedMomentum252Skip21().calc_batch(_panel(rows=rows))).all()


@pytest.mark.parametrize("factor", FACTORS)
def test_corporate_action_price_unit_change_is_neutral(factor):
    panel = _panel()
    original = factor().calc_batch(panel)
    for field in ("close", "preClose", "high", "low"):
        panel[field][100:] *= 0.5
    actual = factor().calc_batch(panel)
    np.testing.assert_allclose(actual, original, rtol=1e-10, atol=1e-12, equal_nan=True)


def test_momentum_excludes_twenty_one_recent_completed_days():
    panel = _panel()
    result = CompletedMomentum252Skip21().calc_batch(panel)
    panel["close"][259:280] *= 2
    changed = CompletedMomentum252Skip21().calc_batch(panel)
    np.testing.assert_array_equal(changed[280], result[280])
    assert not np.array_equal(changed[281], result[281])


@pytest.mark.parametrize("value", [np.nan, np.inf, 0, -1])
def test_amount_missingness_is_explicit(value):
    panel = _panel()
    panel["amount"][100, 0] = value
    result = CompletedAmountImbalance20().calc_batch(panel)
    assert np.isnan(result[101:121, 0]).all()
    assert np.isfinite(result[121, 0])


def test_close_location_zero_range_and_inconsistent_bars_are_missing():
    panel = _panel()
    panel["high"][100, 0] = panel["low"][100, 0]
    panel["close"][100, 1] = panel["high"][100, 1] * 2
    result = CompletedCloseLocation20().calc_batch(panel)
    assert np.isnan(result[101:121, :2]).all()
    assert np.isfinite(result[121]).all()


def test_amount_flat_return_has_zero_signal_without_missingness():
    panel = _panel()
    panel["close"][:] = panel["preClose"]
    result = CompletedAmountImbalance20().calc_batch(panel)
    np.testing.assert_array_equal(result[20:], 0)


@pytest.mark.parametrize("factor", FACTORS)
def test_future_listed_columns_do_not_change_existing_factor_values(factor):
    panel = _panel()
    original = factor().calc_batch(panel)
    expanded = {name: np.column_stack([matrix, np.full((len(matrix), 2), np.nan)]) for name, matrix in panel.items()}
    actual = factor().calc_batch(expanded)
    np.testing.assert_array_equal(actual[:, :3], original)
    assert np.isnan(actual[:, 3:]).all()
