"""similarity._hasher 用模块级 dict 缓存（替代 functools.lru_cache）的单元测试。"""
import numpy as np

from factor_db import similarity as sim


def test_hasher_cached_same_object():
    """相同形状/seed 第二次返回同一缓存对象（不重新生成）。"""
    sim._HASHER_CACHE.clear()
    a = sim._hasher(5, 100, 1024, 42)
    b = sim._hasher(5, 100, 1024, 42)
    assert a is b  # 命中内存缓存
    assert len(sim._HASHER_CACHE) == 1


def test_hasher_deterministic_and_shape():
    sim._HASHER_CACHE.clear()
    bucket, sign = sim._hasher(3, 50, 256, 7)
    assert bucket.shape == (3 * 50,)
    assert sign.shape == (3 * 50,)
    assert bucket.dtype == np.int32
    assert set(np.unique(sign)).issubset({-1.0, 1.0})
    assert bucket.min() >= 0 and bucket.max() < 256


def test_hasher_distinct_keys_distinct_entries():
    """不同 seed / 形状是不同缓存项，且 seed 不同内容不同。"""
    sim._HASHER_CACHE.clear()
    b1, s1 = sim._hasher(4, 20, 128, 1)
    b2, s2 = sim._hasher(4, 20, 128, 2)
    assert len(sim._HASHER_CACHE) == 2
    assert not np.array_equal(b1, b2)
