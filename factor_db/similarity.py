"""因子相似度 / 多样性：全截面 rank 指纹 + NSGA 多目标。

核心思想（又快又准，因子多也扛得住）：
- 把每个因子的每日截面 rank 矩阵压成一个紧凑"指纹向量"，使
      sig_i · sig_j ≈ 两因子的平均每日截面 rank 相关（Pearson on ranks ≈ Spearman），
      sig_i · sig_i ≈ 1。
- 压缩用带符号 feature hashing（count-sketch）：把 (天数×股票数) 维投到 D 维，
  点积无偏，误差 ~ 1/sqrt(D)，调大 D 即更准。存储仅 D 维/因子。
- 相关矩阵 = SIG @ SIG.T（O(N²·D)，一次矩阵乘）；
  多样性_i = 1 - max_{j≠i} corr(i,j)（离最近邻越远越多样；近似重复 → 多样性≈0）。
  每加一个因子，全员多样性都会随相关矩阵刷新。

GA fitness：NSGA-II 非支配排序，目标 = (夏普↑, 多样性↑)。
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_DIM = 16384
DEFAULT_SEED = 20260530
CLONE_CORR = 0.99  # 近克隆阈值：截面 rank 相关 >= 此值即视为行为克隆（选父去重 / 入库闸门 / 库去重统一口径）
DEDUP_CORR = CLONE_CORR



# 按形状缓存 hash 桶：算一次放内存，同形状所有因子复用（模块级 dict，非装饰器缓存）。
_HASHER_CACHE: dict[tuple[int, int, int, int], tuple[np.ndarray, np.ndarray]] = {}


def _hasher(n_days: int, n_stocks: int, dim: int, seed: int):
    """为固定形状预生成 hash 桶下标与符号（同形状所有因子共用 → 点积可比）。"""
    key = (n_days, n_stocks, dim, seed)
    cached = _HASHER_CACHE.get(key)
    if cached is None:
        rng = np.random.default_rng(seed)
        n = n_days * n_stocks
        bucket = rng.integers(0, dim, size=n, dtype=np.int32)
        sign = (rng.integers(0, 2, size=n, dtype=np.int8) * 2 - 1).astype(np.float32)
        cached = (bucket, sign)
        _HASHER_CACHE[key] = cached
    return cached


def signature(rank_mat: np.ndarray, dim: int = DEFAULT_DIM, seed: int = DEFAULT_SEED) -> np.ndarray:
    """由对齐的每日截面 rank 矩阵 (n_days, n_stocks) 生成 D 维指纹。

    rank_mat: scores_to_ranks 产物，有效股 rank∈(0,1]，无效股=0。
    """
    R = np.asarray(rank_mat, dtype=np.float64)
    n_days, n_stocks = R.shape
    valid = R > 0
    m = valid.sum(axis=1).astype(np.float64)                       # 每日有效股数
    ok = m > 1                                                     # 至少 2 只才可标准化
    s1 = np.where(valid, R, 0.0).sum(axis=1)
    s2 = np.where(valid, R * R, 0.0).sum(axis=1)
    mean = np.zeros(n_days); std = np.zeros(n_days)
    mean[ok] = s1[ok] / m[ok]
    var = np.zeros(n_days)
    var[ok] = s2[ok] / m[ok] - mean[ok] ** 2
    std[ok] = np.sqrt(np.maximum(var[ok], 0.0))
    ok &= std > 0

    Z = np.zeros_like(R)
    Z[ok] = (R[ok] - mean[ok, None]) * valid[ok] / std[ok, None]   # 按日标准化，无效→0

    n_eff = int(ok.sum())
    if n_eff == 0:
        return np.zeros(dim, dtype=np.float32)
    scale = np.zeros(n_days)
    scale[ok] = 1.0 / np.sqrt(n_eff * m[ok])                       # 令 sig·sig≈1, sig_i·sig_j≈平均日相关
    V = Z * scale[:, None]

    bucket, sign = _hasher(n_days, n_stocks, dim, seed)
    weights = (sign * V.ravel().astype(np.float32))
    sig = np.bincount(bucket, weights=weights, minlength=dim)[:dim]
    return sig.astype(np.float32)


# ========== 指纹缓存（派生数据，可重建） ==========

_CACHE = Path(__file__).resolve().parent / 'signatures.npz'
_META = Path(__file__).resolve().parent / 'signatures.meta.json'
_CACHE_KEYS = ('dim', 'seed', 'start', 'end', 'n_days', 'n_stocks', 'pool')


def load_cache() -> tuple[list[str], np.ndarray, dict]:
    if not _CACHE.exists():
        return [], np.zeros((0, DEFAULT_DIM), dtype=np.float32), {}
    with np.load(_CACHE, allow_pickle=False) as data:
        names = [str(n) for n in data['names']]
        sigs = data['sigs'].astype(np.float32, copy=False)
    meta = json.loads(_META.read_text(encoding='utf-8')) if _META.exists() else {}
    return names, sigs, meta


def save_cache(names: list[str], sigs: np.ndarray, meta: dict) -> None:
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        _CACHE,
        names=np.asarray(names, dtype='U128'),
        sigs=np.asarray(sigs, dtype=np.float32),
    )
    _META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')


def _same_cache_base(a: dict, b: dict) -> bool:
    return all(a.get(k) == b.get(k) for k in _CACHE_KEYS)


def add_to_cache(name: str, sig: np.ndarray, meta: dict) -> None:
    names, sigs, old_meta = load_cache()
    sig2 = np.asarray(sig, dtype=np.float32).reshape(1, -1)
    if not names or not old_meta or not _same_cache_base(old_meta, meta):
        save_cache([name], sig2, meta)
        return
    if name in names:
        sigs[names.index(name)] = sig2[0]
    else:
        names.append(name)
        sigs = np.vstack([sigs, sig2])
    save_cache(names, sigs, meta)


def cached_matrix() -> tuple[list[str], np.ndarray]:
    """从缓存直接读出 (names, 相关矩阵)。"""
    names, sigs, _ = load_cache()
    return names, correlation_matrix(sigs)


def nearest_in_cache(sig: np.ndarray) -> tuple[Optional[str], float]:
    """该指纹与缓存中现有因子的最大相关 (best_name, best_corr)。零指纹/空缓存返回 (None, 0.0)。"""
    names, sigs, _ = load_cache()
    if len(names) == 0 or float(np.linalg.norm(sig)) < 1e-6:
        return None, 0.0
    norms = np.linalg.norm(sigs, axis=1)
    c = sigs @ sig
    c = np.where(norms < 1e-6, -1.0, c)
    j = int(np.argmax(c))
    return names[j], float(np.clip(c[j], -1.0, 1.0))


def clones_in_cache(sig: np.ndarray, threshold: float = CLONE_CORR) -> list[tuple[str, float]]:
    """缓存中与该指纹截面 rank 相关 >= threshold 的全部因子 [(name, corr), ...]，按相关降序。
    零指纹/空缓存返回 []。"""
    names, sigs, _ = load_cache()
    if len(names) == 0 or float(np.linalg.norm(sig)) < 1e-6:
        return []
    norms = np.linalg.norm(sigs, axis=1)
    c = sigs @ sig
    c = np.where(norms < 1e-6, -1.0, c)
    hits = [(names[i], float(np.clip(c[i], -1.0, 1.0))) for i in range(len(names)) if c[i] >= threshold]
    hits.sort(key=lambda x: x[1], reverse=True)
    return hits


def dedup_representatives(names: list[str], priority: dict[str, float],
                          threshold: float = CLONE_CORR) -> list[str]:
    """对给定因子按指纹聚类去重：每个"行为相同"簇只保留 priority(如夏普) 最高的代表。

    缓存里没有指纹或零指纹的因子原样保留（无法判定相似）。返回保留下来的因子名（无序）。
    """
    cnames, sigs, _ = load_cache()
    idx = {n: i for i, n in enumerate(cnames)}
    kept, rep_sigs = [], []
    for n in sorted(names, key=lambda x: priority.get(x, float('-inf')), reverse=True):
        i = idx.get(n)
        if i is None or float(np.linalg.norm(sigs[i])) < 1e-6:
            kept.append(n)
            continue
        s = sigs[i]
        if any(float(rs @ s) > threshold for rs in rep_sigs):
            continue
        kept.append(n)
        rep_sigs.append(s)
    return kept


def cache_period() -> Optional[tuple[str, str]]:
    """指纹缓存对应的基准回测区间 (start, end)，无缓存返回 None。"""
    _, _, meta = load_cache()
    if meta.get('start') and meta.get('end'):
        return meta['start'], meta['end']
    return None


def correlation_matrix(sigs: np.ndarray) -> np.ndarray:
    """指纹矩阵 (N, D) → 相关矩阵 (N, N)，裁剪到 [-1,1]，对角置 1。"""
    if len(sigs) == 0:
        return np.zeros((0, 0))
    c = sigs @ sigs.T
    np.clip(c, -1.0, 1.0, out=c)
    np.fill_diagonal(c, 1.0)
    return c


def diversity_scores(corr: np.ndarray) -> np.ndarray:
    """多样性_i = 1 - max_{j≠i} corr(i,j)。单因子时为 1。"""
    n = len(corr)
    if n <= 1:
        return np.ones(n)
    c = corr.copy()
    np.fill_diagonal(c, -np.inf)
    nearest = c.max(axis=1)
    return 1.0 - nearest


def diversity_from_signatures(sigs: np.ndarray) -> np.ndarray:
    """由指纹直接算多样性；退化因子（零指纹，输出全 NaN/常数）多样性记为 0（无信号，不应视为多样）。"""
    if len(sigs) == 0:
        return np.zeros(0)
    div = diversity_scores(correlation_matrix(sigs))
    norms = np.linalg.norm(sigs, axis=1)
    return np.where(norms < 1e-6, 0.0, div)


# ========== NSGA-II 多目标 ==========

def _dominates(a, b) -> bool:
    """a 支配 b：每个目标都不差，且至少一个严格更优（均为越大越好）。"""
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def non_dominated_sort(objs: list[tuple]) -> list[list[int]]:
    """快速非支配排序，返回前沿列表 [front0_idx, front1_idx, ...]（front0 = Pareto 前沿）。"""
    n = len(objs)
    S = [[] for _ in range(n)]
    n_dom = [0] * n
    fronts: list[list[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(objs[p], objs[q]):
                S[p].append(q)
            elif _dominates(objs[q], objs[p]):
                n_dom[p] += 1
        if n_dom[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding_distance(front: list[int], objs: list[tuple]) -> dict[int, float]:
    """同一前沿内的拥挤度（边界个体为 inf，越大越稀疏越优先）。"""
    if not front:
        return {}
    dist = {i: 0.0 for i in front}
    n_obj = len(objs[front[0]])
    for m in range(n_obj):
        order = sorted(front, key=lambda i: objs[i][m])
        lo, hi = objs[order[0]][m], objs[order[-1]][m]
        dist[order[0]] = dist[order[-1]] = float('inf')
        span = hi - lo
        if span <= 0:
            continue
        for k in range(1, len(order) - 1):
            dist[order[k]] += (objs[order[k + 1]][m] - objs[order[k - 1]][m]) / span
    return dist


def nsga_select(objs: list[tuple], k: int) -> list[int]:
    """按 NSGA-II（前沿层级 + 拥挤度）选出 k 个个体的下标。"""
    if k >= len(objs):
        return list(range(len(objs)))
    chosen: list[int] = []
    for front in non_dominated_sort(objs):
        if len(chosen) + len(front) <= k:
            chosen.extend(front)
        else:
            cd = crowding_distance(front, objs)
            front_sorted = sorted(front, key=lambda i: cd[i], reverse=True)
            chosen.extend(front_sorted[: k - len(chosen)])
            break
    return chosen
