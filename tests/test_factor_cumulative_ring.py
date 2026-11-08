"""Exact pre-refactor NumPy arithmetic is the oracle, including float32."""
from __future__ import annotations
import subprocess
import sys
import numpy as np
import pytest
from factor_db.factors.VolumeCV import VolumeCV
from factor_db.factors.AmountBasedSmallCap import AmountBasedSmallCap
from factor.library.completed_windows import completed_finite_window_sum


def _numpy_completed_sum(
    values: np.ndarray, window: int, skip: int = 0, *, min_count: int | None = None,
) -> np.ndarray:
    """The single prefix/suffix kernel for both full-width and tiled windows."""
    required = window if min_count is None else min_count
    rows, stocks = values.shape
    result = np.full((rows, stocks), np.nan, dtype=np.float64)
    first = window + skip
    if rows <= first:
        return result
    # Prefix/suffix windows avoid subtracting a long-history cumulative sum:
    # an old large amount cannot erase a later small but valid window. Blocks
    # keep the working arrays in cache, while each operation spans all stocks.
    previous = values[:window]
    for end in range(window, rows - skip, window):
        length = min(window, rows - skip - end)
        prior_valid = np.isfinite(previous)
        suffix = np.cumsum(np.where(prior_valid, previous, 0.0)[::-1], axis=0)[::-1]
        suffix_count = np.cumsum(prior_valid[::-1], axis=0, dtype=np.int32)[::-1]
        current = values[end : end + length]
        current_valid = np.isfinite(current)
        prefix = np.empty((length, stocks), dtype=np.float64)
        prefix_count = np.empty((length, stocks), dtype=np.int32)
        prefix[0] = 0.0
        prefix_count[0] = 0
        if length > 1:
            np.cumsum(np.where(current_valid[:-1], current[:-1], 0.0), axis=0, out=prefix[1:])
            np.cumsum(current_valid[:-1], axis=0, dtype=np.int32, out=prefix_count[1:])
        complete = suffix_count[:length] + prefix_count >= required
        result[end + skip : end + skip + length] = np.where(complete, suffix[:length] + prefix, np.nan)
        previous = values[end : end + window]
    return result


def _old_volume(values, window=20):
    known = np.empty_like(values)
    known[0] = np.nan
    known[1:] = values[:-1]
    count = np.cumsum(~np.isnan(known), axis=0).astype(float)
    summed = np.cumsum(np.where(np.isnan(known), 0.0, known), axis=0)
    squared = np.cumsum(np.where(np.isnan(known), 0.0, known * known), axis=0)
    cv = np.empty_like(known, dtype=float)
    cv[:window] = np.nan
    n = count[window:] - count[:-window]
    mean = (summed[window:] - summed[:-window]) / n
    mean_sq = (squared[window:] - squared[:-window]) / n
    variance = mean_sq - mean * mean
    cv[window:] = np.sqrt(np.maximum(variance, 0.0)) / mean
    return np.where(~np.isnan(cv), -cv, np.nan)


def _old_amount(values, window=60):
    known = np.empty_like(values)
    known[0] = np.nan
    known[1:] = values[:-1]
    summed = np.cumsum(np.where(np.isnan(known), 0.0, known), axis=0)
    count = np.cumsum(~np.isnan(known), axis=0).astype(float)
    average = np.empty_like(values, dtype=float)
    average[:window] = summed[:window] / count[:window]
    average[window:] = (summed[window:] - summed[:-window]) / (count[window:] - count[:-window])
    average /= 1e8
    score = 100 * np.exp(-(average / 5))
    return np.where(~np.isnan(average), score, np.nan)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('rows', [0, 1, 2, 19, 20, 21, 59, 60, 61, 151, 523])
@pytest.mark.parametrize('implementation,oracle,field', [
    (VolumeCV, _old_volume, 'volume'), (AmountBasedSmallCap, _old_amount, 'amount'),
])
def test_cumulative_factor_matches_original_bytes(dtype, rows, implementation, oracle, field):
    rng = np.random.default_rng(rows + 20260917)
    values = rng.normal(10, 100, (rows, 19)).astype(dtype)
    if rows:
        values[:, 0] = np.nan
        values[:, 1] = 0
        values[:, 2] = -0.0
        values[0, 3] = np.finfo(dtype).max
        values[0, 4] = np.inf
        values[-1, 5] = -np.inf
        values[-1, 6] = np.finfo(dtype).tiny
        values[::11, 7] = np.nan
        values[::17, 8] = -np.inf
        values[::13, 9] = np.inf
    before = values.tobytes()
    with np.errstate(all='ignore'):
        if rows == 0:
            with pytest.raises(IndexError):
                oracle(values)
            with pytest.raises(IndexError):
                implementation().calc_batch({field: values})
            return
        expected = oracle(values)
        actual = implementation().calc_batch({field: values})
    assert expected.dtype == actual.dtype
    assert expected.tobytes() == actual.tobytes()
    assert values.tobytes() == before


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('implementation,oracle,field', [
    (VolumeCV, _old_volume, 'volume'), (AmountBasedSmallCap, _old_amount, 'amount'),
])
def test_readonly_noncontiguous_inputs_and_t_day_causality(dtype, implementation, oracle, field):
    rng = np.random.default_rng(771)
    values = rng.lognormal(5, 2, (201, 44)).astype(dtype)[:, ::2]
    values.flags.writeable = False
    with np.errstate(all='ignore'):
        actual = implementation().calc_batch({field: values})
        assert actual.tobytes() == oracle(values).tobytes()
    changed = values.copy()
    changed[103:] *= .1
    np.testing.assert_array_equal(implementation().calc_batch({field: changed})[:104], actual[:104])


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('rows,stocks', [(0, 137), (19, 137), (20, 137), (21, 137),
                                        (253, 137), (513, 137), (513, 0), (513, 1)])
@pytest.mark.parametrize('window,skip,minimum', [(20, 0, None), (231, 21, 185)])
def test_compiled_windows_preserve_numpy_arithmetic_bytes(dtype, rows, stocks, window, skip, minimum):
    rng = np.random.default_rng(78)
    values = rng.normal(size=(rows, stocks)).astype(dtype)
    if rows and stocks:
        values[0, 0] = np.finfo(dtype).max / 2
        values[-1, -1] = np.nan
        if stocks > 2:
            values[::11, 1] = np.inf
            values[::17, 2] = -np.inf
    before = values.tobytes()
    with np.errstate(all='ignore'):
        expected = _numpy_completed_sum(values, window, skip, min_count=minimum)
        actual = completed_finite_window_sum(values, window, skip, min_count=minimum)
    assert actual.tobytes() == expected.tobytes()
    assert values.tobytes() == before


@pytest.mark.parametrize('name,field', [('VolumeCV', 'volume'), ('AmountBasedSmallCap', 'amount')])
def test_factor_import_and_direct_use_without_registry_bootstrap(name, field):
    command = (
        f'from factor_db.factors.{name} import {name}; '
        'import numpy as np; '
        f'value = {name}().calc_batch({{{field!r}: np.ones((80, 3), dtype=np.float32)}}); '
        'assert value.shape == (80, 3)'
    )
    subprocess.run([sys.executable, '-c', command], check=True, capture_output=True, text=True)


@pytest.mark.parametrize('dtype', [np.float16, np.int64, np.bool_, np.dtype('>f4'), np.dtype('>f8')])
def test_finite_window_rejects_unsupported_accumulator_precision(dtype):
    with pytest.raises(TypeError, match='native float32 or float64'):
        completed_finite_window_sum(np.ones((24, 3), dtype=dtype), 20)


@pytest.mark.parametrize('dtype,integer', [(np.float32, np.uint32), (np.float64, np.uint64)])
@pytest.mark.parametrize('window,skip,minimum', [(2, 0, 1), (20, 0, 20), (231, 21, 185)])
def test_finite_window_ieee_bit_patterns_preserve_overflow_and_signed_nan(dtype, integer, window, skip, minimum):
    rng = np.random.default_rng(541)
    for _ in range(6):
        values = rng.integers(0, np.iinfo(integer).max, (2 * window + skip + 7, 17), dtype=integer).view(dtype)
        values[:2, 0] = np.finfo(dtype).max
        values[2:4, 0] = -np.finfo(dtype).max
        values.flags.writeable = False
        with np.errstate(all='ignore'):
            expected = _numpy_completed_sum(values, window, skip, min_count=minimum)
            actual = completed_finite_window_sum(values, window, skip, min_count=minimum)
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_finite_window_noncontiguous_input_is_causal_and_readonly(dtype):
    values = np.random.default_rng(442).normal(size=(517, 26)).astype(dtype)[:, ::2]
    values.flags.writeable = False
    before = values.tobytes()
    expected = _numpy_completed_sum(values, 231, 21, min_count=185)
    actual = completed_finite_window_sum(values, 231, 21, min_count=185)
    assert actual.tobytes() == expected.tobytes()
    assert values.tobytes() == before
    changed = values.copy()
    changed[310:] = 2
    later = completed_finite_window_sum(changed, 231, 21, min_count=185)
    assert later[:332].tobytes() == actual[:332].tobytes()
