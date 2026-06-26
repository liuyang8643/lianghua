"""factor_db.similarity 单元测试：指纹点积≈全截面相关、多样性、NSGA 多目标。"""
import numpy as np
import pytest

from factor_db import similarity


def _rank_mat_from_scores(scores: np.ndarray) -> np.ndarray:
    """把原始分数按每日截面转成 rank∈(0,1]（与 core.scoring.scores_to_ranks 同口径，无效→0）。"""
    from core.scoring import scores_to_ranks
    return scores_to_ranks(scores.astype(np.float32))


def test_signature_dot_approximates_correlation():
    rng = np.random.default_rng(0)
    n_days, n_stocks = 200, 400
    a = rng.standard_normal((n_days, n_stocks))
    # b 与 a 截面强相关，c 独立
    b = a + 0.1 * rng.standard_normal((n_days, n_stocks))
    c = rng.standard_normal((n_days, n_stocks))
    Ra, Rb, Rc = (_rank_mat_from_scores(x) for x in (a, b, c))

    sig_a = similarity.signature(Ra, dim=32768, seed=1)
    sig_b = similarity.signature(Rb, dim=32768, seed=1)
    sig_c = similarity.signature(Rc, dim=32768, seed=1)

    assert sig_a @ sig_a == pytest.approx(1.0, abs=0.05)        # 自相关≈1
    assert sig_a @ sig_b > 0.7                                  # 强相关
    assert abs(sig_a @ sig_c) < 0.2                             # 近似不相关


def test_correlation_matrix_and_diversity():
    rng = np.random.default_rng(2)
    base = _rank_mat_from_scores(rng.standard_normal((150, 300)))
    dup = base.copy()                                          # 完全重复
    other = _rank_mat_from_scores(rng.standard_normal((150, 300)))
    sigs = np.array([similarity.signature(m, dim=32768, seed=3) for m in (base, dup, other)])

    corr = similarity.correlation_matrix(sigs)
    assert corr.shape == (3, 3)
    assert np.allclose(np.diag(corr), 1.0)
    assert corr[0, 1] > 0.95                                   # base 与 dup 几乎相同

    div = similarity.diversity_scores(corr)
    assert div[0] < 0.1 and div[1] < 0.1                      # 互为重复 → 多样性≈0
    assert div[2] > div[0]                                     # 独立因子更多样


def test_diversity_edge_cases():
    assert similarity.diversity_scores(np.zeros((0, 0))).tolist() == []
    assert similarity.diversity_scores(np.ones((1, 1))).tolist() == [1.0]


def test_non_dominated_sort_fronts():
    # 目标均越大越好。A 支配 B；C 与 A 互不支配（各占一目标）
    objs = [(1.0, 1.0), (0.5, 0.5), (0.2, 2.0), (0.0, 0.0)]
    fronts = similarity.non_dominated_sort(objs)
    assert set(fronts[0]) == {0, 2}        # Pareto 前沿
    assert 1 in fronts[1]
    assert 3 in fronts[-1]


def test_nsga_select_prefers_front0():
    objs = [(1.0, 1.0), (0.2, 2.0), (0.5, 0.5), (0.0, 0.0)]
    chosen = similarity.nsga_select(objs, 2)
    assert set(chosen) == {0, 1}           # 选 Pareto 前沿两点


def test_crowding_distance_boundary_inf():
    objs = [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]
    cd = similarity.crowding_distance([0, 1, 2], objs)
    assert cd[0] == float('inf') and cd[2] == float('inf')   # 边界
    assert cd[1] < float('inf')


def test_cache_roundtrip_and_add(tmp_path, monkeypatch):
    monkeypatch.setattr(similarity, '_CACHE', tmp_path / 'sig.npz')
    monkeypatch.setattr(similarity, '_META', tmp_path / 'sig.meta.json')
    meta = {'dim': 8, 'seed': 1, 'start': 's', 'end': 'e', 'n_days': 2, 'n_stocks': 3, 'pool': 'all_A'}
    similarity.add_to_cache('F1', np.arange(8, dtype=np.float32), meta)
    similarity.add_to_cache('F2', np.ones(8, dtype=np.float32), meta)
    names, sigs, m = similarity.load_cache()
    assert names == ['F1', 'F2']
    assert sigs.shape == (2, 8)
    assert m['dim'] == 8

    # 基底变化 → 缓存重置
    similarity.add_to_cache('F3', np.zeros(4, dtype=np.float32), {**meta, 'dim': 4, 'n_days': 9})
    names2, sigs2, _ = similarity.load_cache()
    assert names2 == ['F3'] and sigs2.shape == (1, 4)


def test_nearest_in_cache_and_dedup(tmp_path, monkeypatch):
    monkeypatch.setattr(similarity, '_CACHE', tmp_path / 'sig.npz')
    monkeypatch.setattr(similarity, '_META', tmp_path / 'sig.meta.json')
    meta = {'dim': 3, 'seed': 1, 'start': 's', 'end': 'e', 'n_days': 1, 'n_stocks': 3, 'pool': 'p'}
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    a2 = np.array([0.999, 0.044, 0.0], dtype=np.float32)   # 与 a 几乎共线（corr≈1）
    b = np.array([0.0, 1.0, 0.0], dtype=np.float32)        # 与 a 正交
    similarity.add_to_cache('A', a, meta)
    similarity.add_to_cache('B', b, meta)

    name, corr = similarity.nearest_in_cache(a2)
    assert name == 'A' and corr > similarity.DEDUP_CORR
    name2, corr2 = similarity.nearest_in_cache(np.array([0.0, 0.0, 1.0], dtype=np.float32))
    assert corr2 < 0.5                                     # 与 A、B 都不相关

    # 零指纹不参与比较
    assert similarity.nearest_in_cache(np.zeros(3, dtype=np.float32)) == (None, 0.0)

    # dedup：A 与 A2 同簇（留夏普高的 A2），B 独立 → 保留 {A2, B}
    similarity.add_to_cache('A2', a2, meta)
    keep = set(similarity.dedup_representatives(['A', 'A2', 'B'], {'A': 0.5, 'A2': 0.9, 'B': 0.3}))
    assert keep == {'A2', 'B'}
