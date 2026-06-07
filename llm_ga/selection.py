"""父代采样：NSGA-II 多目标（夏普↑ + 多样性↑）非支配排序选父，供交叉/变异。

多样性 = 1 - 与最近邻的全截面 rank 相关（来自 similarity 指纹缓存），是相对量：
每新增一个因子，全员多样性都会随之刷新。只有夏普与多样性都不差且至少一个更优的个体
才会支配别人（落在更靠前的 Pareto 前沿），优先被选为父代。

冷启动（库内尚无评分因子或无指纹缓存）时退化为随机采样。只挑选自身能通过 guard 的因子，
使子代更可能保持合法（不引入禁止字段）。
"""
import random

from factor_db import db, records, similarity
from llm_ga import guard
from llm_ga.config import REPO_ROOT


def _read_code(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding='utf-8')


def _candidate_pool(param_cap: int, core_factors: list[str] | None = None) -> list[dict]:
    pool = []
    for f in db.list_factors():
        if f['status'] not in ('passed', 'seed', 'active'):
            continue
        if core_factors is not None and f['name'] not in core_factors:
            continue
        try:
            code = _read_code(f['file_path'])
        except OSError:
            continue
        # 人工种子因子（active/seed）已人工审查，无需过自动 guard；GA 产出（passed）需过闸
        if f['status'] == 'passed':
            ok, _, _ = guard.check(code, param_cap)
            if not ok:
                continue
        f = dict(f)
        f['code'] = code
        pool.append(f)
    return pool


def _sharpe_of(f: dict) -> float | None:
    """因子夏普：优先 db 摘要，缺失时回退 factor_runs 中【与指纹同基准区间】的回测（保证可比）。"""
    if f.get('train_sharpe') is not None:
        return f['train_sharpe']
    period = similarity.cache_period()
    run = records.get_run(f['name'], *period) if period else None
    run = run or records.get_latest_run(f['name'])
    return run['sharpe'] if run else None


def _diversity_map(names: list[str]) -> dict[str, float]:
    """对给定因子集合，从指纹缓存算每个因子的多样性（1 - 最近邻相关）。

    缓存里没有指纹的因子记为 1.0（视为新颖，不惩罚）；退化因子（零指纹）记为 0。
    """
    cache_names, sigs, _ = similarity.load_cache()
    if len(cache_names) < 2:
        return {n: 1.0 for n in names}
    idx = {n: i for i, n in enumerate(cache_names)}
    present = [n for n in names if n in idx]
    if len(present) < 2:
        return {n: 1.0 for n in names}
    sub = sigs[[idx[n] for n in present]]
    div = similarity.diversity_from_signatures(sub)
    out = {n: 1.0 for n in names}
    out.update({n: float(d) for n, d in zip(present, div)})
    return out


def top_factors(n: int, param_cap: int, core_factors: list[str] | None = None) -> list[dict]:
    """全库（能过 guard 的）按夏普降序取前 n 个（含 code），用作变异的"灵感因子"。"""
    pool = _candidate_pool(param_cap, core_factors)
    scored = [(f, _sharpe_of(f)) for f in pool]
    scored = [(f, s) for f, s in scored if s is not None]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [f for f, _ in scored[:n]]


def best_factor_name(param_cap: int, core_factors: list[str] | None = None) -> str | None:
    """全库（能过 guard 的）按夏普最优的因子名，用作初始 elite。"""
    pool = _candidate_pool(param_cap, core_factors)
    scored = [(f['name'], _sharpe_of(f)) for f in pool]
    scored = [(n, s) for n, s in scored if s is not None]
    return max(scored, key=lambda x: x[1])[0] if scored else None


def top_half_by_nsga(param_cap: int, core_factors: list[str] | None = None) -> list[dict]:
    """全库能过 guard 的因子，先按指纹做行为去重（克隆只留夏普最高代表），再按
    NSGA(夏普↑+多样性↑) 非支配排序取前 50%。去重避免 top50% 被同质克隆占满。"""
    pool = _candidate_pool(param_cap, core_factors)
    sharpes = {f['name']: _sharpe_of(f) for f in pool}
    scored = [f for f in pool if sharpes[f['name']] is not None]
    if not scored:
        return pool
    keep = set(similarity.dedup_representatives(
        [f['name'] for f in scored], {n: s for n, s in sharpes.items() if s is not None}))
    scored = [f for f in scored if f['name'] in keep]
    div = _diversity_map([f['name'] for f in scored])
    objs = [(sharpes[f['name']], div[f['name']]) for f in scored]
    k = max(1, len(scored) // 2)
    idx = similarity.nsga_select(objs, k)
    return [scored[i] for i in idx]


def select_parents(n_random: int, param_cap: int, rng: random.Random,
                   elite_name: str | None = None,
                   core_factors: list[str] | None = None) -> list[dict]:
    """选父代：n_elite 个上一轮最优(elite_name) + n_random 个从全库 top50% 随机挑。

    父代均来自现有库（已回测、已入库），不需要再过 LLM / 再回测。冷启动（无评分）退化为随机。
    """
    pool = _candidate_pool(param_cap, core_factors)
    if not pool:
        return []
    by_name = {f['name']: f for f in pool}
    elite = by_name.get(elite_name)

    top_half = top_half_by_nsga(param_cap, core_factors)
    if not any(_sharpe_of(f) is not None for f in pool):  # 冷启动
        top_half = pool
    cand = [f for f in top_half if f['name'] != elite_name]
    picks = rng.sample(cand, min(n_random, len(cand))) if cand else []

    parents = ([elite] if elite else []) + picks
    return parents
