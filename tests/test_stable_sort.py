"""Independent NumPy oracle for the shared numerical ordering primitive."""
import numpy as np
import pytest

from utils.stable_sort import stable_radix_order


@pytest.mark.parametrize("count", [0, 1, 17, 1000])
@pytest.mark.parametrize("duplicates", [False, True])
def test_uint64_key_order_is_stable_for_subsets_and_reused_workspaces(count, duplicates):
    rng = np.random.default_rng(20260917)
    keys = rng.integers(0, np.iinfo(np.uint64).max, 1000, dtype=np.uint64)
    keys[:4] = [0, 2**63 - 1, 2**63, np.iinfo(np.uint64).max]
    if duplicates:
        keys %= np.uint64(5)
    original_keys = keys.copy()
    order = np.empty(1000, dtype=np.intp)
    scratch = np.empty_like(order)
    buckets = np.empty(256, dtype=np.intp)
    for _ in range(3):
        order[:] = rng.permutation(1000)
        selected = order[:count].copy()
        expected = selected[np.argsort(keys[selected], kind="stable")]
        order, scratch = stable_radix_order(keys, order, scratch, count, buckets)
        np.testing.assert_array_equal(order[:count], expected)
        np.testing.assert_array_equal(keys, original_keys)
