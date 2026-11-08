"""Stable uint64-key ordering shared by factor ranks and stock selection."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False, parallel=False)
def stable_radix_order(keys, order, scratch, count, buckets):
    """Sort ``order[:count]`` by ascending keys, retaining equal-key order.

    Callers provide validated uint64 keys, valid integer indices in ``order``,
    a same-sized integer scratch array, and 256 integer buckets. The two index
    workspaces are mutated and returned as (sorted, scratch) for reuse; their
    tails beyond ``count`` have no defined value. No workspaces are allocated.
    Key construction and all domain/input validation remain with callers.
    """
    if count < 2:
        return order, scratch
    first = keys[order[0]]
    varying = np.uint64(0)
    for index in range(1, count):
        varying |= keys[order[index]] ^ first
    for byte in range(8):
        shift = np.uint64(byte * 8)
        if ((varying >> shift) & np.uint64(255)) == 0:
            continue
        buckets[:] = 0
        for position in range(count):
            digit = (keys[order[position]] >> shift) & np.uint64(255)
            buckets[digit] += 1
        offset = 0
        for digit in range(256):
            size = buckets[digit]
            buckets[digit] = offset
            offset += size
        for position in range(count):
            index = order[position]
            digit = (keys[index] >> shift) & np.uint64(255)
            scratch[buckets[digit]] = index
            buckets[digit] += 1
        order, scratch = scratch, order
    return order, scratch


__all__ = ["stable_radix_order"]
