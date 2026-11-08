"""Exact oracles for the single rank and final-cache writer.

The references deliberately retain the former NumPy expressions rather than
reimplementing the radix algorithm. Source precision is part of the contract.
"""

import numpy as np
import pytest

from factor.compute import _score_cache_kernel, scores_to_ranks


def _numpy_ranks(values, membership=None, binary=False):
    if binary:
        valid = np.isfinite(values)
        if membership is not None:
            valid &= membership
        if np.any(valid & (values != 0.0) & (values != 1.0)):
            raise ValueError("binary factor scores must be 0 or 1 when finite")
        return np.where(valid, values, 0.0).astype(np.float32)
    result = np.zeros(values.shape, dtype=np.float32)
    for row_index, row in enumerate(values):
        valid = np.isfinite(row)
        if membership is not None:
            valid &= membership[row_index]
        columns = np.flatnonzero(valid)
        count = int(valid.sum())
        if count == 0:
            continue
        order = np.argsort(row[valid])[::-1]
        sorted_values = row[columns[order]]
        starts = np.flatnonzero(np.r_[True, sorted_values[1:] != sorted_values[:-1]])
        stops = np.r_[starts[1:], count]
        positions = np.repeat((starts + stops - 1) * 0.5, stops - starts).astype(np.float32)
        result[row_index, columns[order]] = 1.0 - positions / count
    return result


def _values(dtype):
    rng = np.random.default_rng(294)
    values = rng.integers(0, 17, (4, 29)).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        limits = np.finfo(dtype)
        values[0, :10] = [np.nan, -np.nan, np.inf, -np.inf, -0.0, 0.0,
                          limits.max, -limits.max, limits.tiny, -limits.tiny]
        values[1, :4] = [1, np.nextafter(dtype(1), dtype(2)),
                         np.nextafter(dtype(1), dtype(0)), 1]
        values[3] = np.nan
    elif dtype == np.int64:
        values[0, :8] = [-2**63, 2**63 - 1, 2**53, 2**53 + 1,
                         -2**53, -2**53 - 1, 2**63 - 2, -2**63 + 1]
    elif dtype == np.uint64:
        values[0, :5] = [2**64 - 1, 2**64 - 2, 2**53, 2**53 + 1, 0]
    return values


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int32, np.int64, np.uint64, np.bool_])
@pytest.mark.parametrize("use_membership", [False, True])
def test_public_ranks_preserve_source_precision_and_numpy_denominator(dtype, use_membership):
    values = _values(dtype)
    member = np.random.default_rng(19).random(values.shape) < 0.65 if use_membership else None
    values.flags.writeable = False
    original = values.tobytes()
    expected = _numpy_ranks(values, member)
    actual = scores_to_ranks(values, membership=member)
    assert actual.dtype == np.float32
    assert actual.tobytes() == expected.tobytes()
    assert values.tobytes() == original


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.uint64])
def test_noncontiguous_input_and_membership_preserve_ties(dtype):
    values = _values(dtype).T[::2, ::-1]
    member = (np.arange(values.size).reshape(values.shape) % 3 != 0)[:, ::-1]
    original = values.tobytes()
    actual = scores_to_ranks(values, membership=member)
    assert actual.tobytes() == _numpy_ranks(values, member).tobytes()
    assert values.tobytes() == original


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int32, np.int64, np.uint64])
def test_non_native_input_uses_same_rank_core_without_losing_precision(dtype):
    source = _values(dtype)
    values = source.astype(source.dtype.newbyteorder("S"))
    values.flags.writeable = False
    member = np.random.default_rng(23).random(values.shape) < 0.7
    original = values.tobytes()
    actual = scores_to_ranks(values, membership=member)
    assert actual.tobytes() == _numpy_ranks(values, member).tobytes()
    assert values.tobytes() == original


@pytest.mark.parametrize("shape", [(0, 0), (0, 4), (3, 0), (1, 1)])
@pytest.mark.parametrize("binary", [False, True])
def test_empty_axes_and_singleton(shape, binary):
    values = np.ones(shape, dtype=np.float64)
    actual = scores_to_ranks(values, score_semantics="binary" if binary else "continuous")
    assert actual.tobytes() == _numpy_ranks(values, binary=binary).tobytes()
    assert actual.shape == shape


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_binary_validity_is_independent_of_pit_and_preserves_signed_zero(dtype):
    values = np.array([[0, -0.0, 1, 0.3, np.nan, np.inf, -np.inf]], dtype=dtype)
    member = np.array([[True, True, True, False, True, True, True]])
    actual = scores_to_ranks(values, membership=member, score_semantics="binary")
    assert actual.tobytes() == _numpy_ranks(values, member, binary=True).tobytes()
    member[0, 3] = True
    with pytest.raises(ValueError, match="0 or 1"):
        scores_to_ranks(values, membership=member, score_semantics="binary")


@pytest.mark.parametrize("binary", [False, True])
def test_fused_writer_matches_old_numpy_bytes_in_final_strided_layout(binary):
    rng = np.random.default_rng(73)
    values = rng.integers(0, 2, (7, 29)).astype(np.float64) if binary else _values(np.float64)
    # Distinct signed quiet NaN payloads must survive raw float32 conversion.
    payloads = np.array([0x7FF8000020000000, 0xFFF8000040000000], dtype=np.uint64).view(np.float64)
    values[0, :6] = [payloads[0], payloads[1], np.inf, -np.inf, 0, -0.0]
    member = rng.random(values.shape) < 0.6
    shape = (values.shape[0], 3, values.shape[1])
    raw = np.full(shape, np.float32(17.25))
    validity = np.ones(shape, dtype=np.bool_)
    ranks = np.full(shape, np.float32(23.75))
    expected_raw = np.empty(values.shape, np.float32)
    finite = np.isfinite(values)
    limit = np.float64(np.finfo(np.float32).max)
    np.clip(values, -limit, limit, out=expected_raw)
    np.copyto(expected_raw, values, where=~finite, casting="unsafe")
    expected_ranks = _numpy_ranks(values, member, binary=binary)
    values.flags.writeable = False
    _score_cache_kernel(values, member, raw[:, 1], validity[:, 1], ranks[:, 1], binary, 0)
    assert raw[:, 1].tobytes() == expected_raw.tobytes()
    assert validity[:, 1].tobytes() == finite.tobytes()
    assert ranks[:, 1].tobytes() == expected_ranks.tobytes()
    for index in (0, 2):
        assert np.all(raw[:, index] == np.float32(17.25))
        assert np.all(validity[:, index])
        assert np.all(ranks[:, index] == np.float32(23.75))


def test_appending_nonmembers_does_not_change_existing_ranks():
    values = _values(np.float64)
    member = np.ones(values.shape, dtype=np.bool_)
    expected = scores_to_ranks(values, membership=member)
    future = np.full((values.shape[0], 17), 1e200)
    actual = scores_to_ranks(np.concatenate((values, future), axis=1),
                             membership=np.concatenate((member, np.zeros_like(future, dtype=bool)), axis=1))
    assert actual[:, :values.shape[1]].tobytes() == expected.tobytes()
    assert np.all(actual[:, values.shape[1]:] == 0)
