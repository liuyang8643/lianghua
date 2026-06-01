"""进化主循环（MadEvolve 式）。

每代（population = 父代 + 子代，默认 5 + 5）：
  1. 选父代：n_elite 个上一轮最优保留 + 其余从全库 top50%(NSGA: 夏普+多样性) 随机挑。
     父代均为库中现有因子，【不重新过 LLM、不重新回测】。
  2. 对每个子代随机决定走【交叉】(综合多父代) 或【变异】(对单父代做定向修改，附灵感因子)。
  3. 并发调用 LLM 产出子代（I/O 并发的子进程，不违反 CPU 多进程红线）。
  4. 子代逐个过 guard / LLM-verify / 连续性闸门 → 回测 → 入库 + 明细 + 指纹。
  5. elite 更新为本代最优子代，传入下一代。
"""
import ast
import json
import random
import shutil
from concurrent.futures import ThreadPoolExecutor

from core.runtime import load_runtime_npz
from factor_db import db, dedup_library, records, similarity
from llm_ga import agent, evaluator, guard, selection, verify
from llm_ga.config import FACTORS_DIR, REPO_ROOT, RunConfig

_PRELOAD_LOOKBACK = 750  # 预加载 NPZ 的回看交易日数（≥ 首个回测日前可用历史 → 与逐因子加载结果一致）


def _parse_thesis(code: str) -> str:
    """从因子代码里提取模块级 `__thesis__ = "..."`（一句话思路），缺省返回空串。"""
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == '__thesis__':
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        return node.value.value.strip()
    return ''


def _build_jobs(gen: int, cfg: RunConfig, parents: list[dict], inspirations: list[dict],
                rng: random.Random) -> list[dict]:
    """为本代 n_offspring 个子代各生成一个任务（随机交叉/变异 + 各自父代 + 独立 scratch tag）。"""
    jobs = []
    for i in range(cfg.n_offspring):
        op = 'crossover' if (len(parents) >= 2 and rng.random() < cfg.crossover_ratio) else 'mutation'
        tag = f'{cfg.run_id}_g{gen}_{i}'   # 含 run_id → 跨 run 全局唯一，避免与历史因子重名
        name = f'Factor_{tag}'
        if op == 'crossover':
            k = min(cfg.n_parents_crossover, len(parents))
            sel = rng.sample(parents, k)
            insp = []
        else:
            sel = [rng.choice(parents)]
            insp = [f for f in inspirations if f['name'] != sel[0]['name']]
        jobs.append({'name': name, 'op': op, 'parents': sel, 'inspirations': insp, 'tag': tag})
    return jobs


def _propose_job(job: dict, cfg: RunConfig) -> tuple[dict, object]:
    """单个子代的 LLM 产因子（在线程池中并发执行）。返回 (job, path|None)。"""
    try:
        written = agent.propose(job['op'], job['parents'], [job['name']], cfg.model,
                                cfg.param_cap, job['tag'], job['inspirations'])
        return job, written.get(job['name'])
    except Exception as e:
        print(f'  ! {job["name"]}: LLM 产因子失败 ({type(e).__name__}: {e})')
        return job, None


def run_generation(gen: int, cfg: RunConfig, dates, stocks, rng: random.Random,
                   elite_name: str | None, panel=None) -> list[dict]:
    n_random = max(0, cfg.n_parents - cfg.n_elite)
    parents = selection.select_parents(n_random, cfg.param_cap, rng, elite_name)
    if not parents:
        print(f'[gen {gen}] 无可用父代，跳过')
        return []
    inspirations = selection.top_factors(cfg.n_inspirations, cfg.param_cap)

    print(f'[gen {gen}] 父代={[p["name"] for p in parents]} (elite={elite_name})')
    jobs = _build_jobs(gen, cfg, parents, inspirations, rng)
    for j in jobs:
        print(f'  · {j["name"]} <- {j["op"]} {[p["name"] for p in j["parents"]]}')

    # 阶段1：并发产因子（子进程 I/O 并发）
    with ThreadPoolExecutor(max_workers=cfg.concurrency) as ex:
        produced = list(ex.map(lambda j: _propose_job(j, cfg), jobs))

    # 阶段2：guard + sha 去重（同步、廉价）→ 候选；不通过的直接出局，避免浪费后续 verify/回测
    results, candidates = [], []
    existing_sha = {f['code_sha256'] for f in db.list_factors()}
    for job, path in produced:
        name = job['name']
        if path is None:
            results.append({'name': name, 'status': 'rejected', 'reason': 'no_output', 'sharpe': None})
            continue
        code = path.read_text(encoding='utf-8')
        ok, n_params, reason = guard.check(code, cfg.param_cap)
        if not ok:
            print(f'  - {name}: guard 拒绝 ({reason})')
            results.append({'name': name, 'status': 'rejected', 'reason': reason, 'sharpe': None})
            continue
        sha = db.file_sha256(path)
        if sha in existing_sha:
            print(f'  - {name}: 重复因子（sha 命中），跳过')
            continue
        existing_sha.add(sha)
        candidates.append({'job': job, 'path': path, 'code': code, 'sha': sha, 'n_params': n_params})

    # 阶段3：并发 LLM verify 红线审查（之前串行是大瓶颈）
    if cfg.llm_verify and candidates:
        with ThreadPoolExecutor(max_workers=cfg.concurrency) as ex:
            verdicts = list(ex.map(lambda c: verify.review(c['code'], cfg.verify_model), candidates))
    else:
        verdicts = [(True, '')] * len(candidates)

    # 阶段4：串行回测 + 入库（纯 numpy 计算遵守 CPU 单进程红线；复用预加载 NPZ 面板）
    for c, (v_ok, v_reason) in zip(candidates, verdicts):
        job, path, name = c['job'], c['path'], c['job']['name']
        code, n_params, sha = c['code'], c['n_params'], c['sha']
        if not v_ok:
            print(f'  - {name}: LLM verify 否决 ({v_reason})')
            results.append({'name': name, 'status': 'rejected', 'reason': f'verify: {v_reason}',
                            'sharpe': None})
            continue
        try:
            cls = evaluator.load_factor_class(path, name)
            metrics = evaluator.evaluate_detailed(cls, name, dates, stocks, cfg.buy_n,
                                                  want_sig=True, data=panel)
        except Exception as e:  # 动态加载/执行/连续性失败归类为 reject
            print(f'  - {name}: 执行失败，拒绝 ({type(e).__name__}: {e})')
            results.append({'name': name, 'status': 'rejected', 'reason': str(e), 'sharpe': None})
            continue

        # append 前相似性闸门：与已有因子截面 rank 近克隆（corr>=CLONE_CORR=0.99）时比夏普——
        # 新因子夏普更高则【覆盖】（删掉被克隆的旧因子全簇，新因子入库做代表）；
        # 不更高则拒绝新因子（保留更优的旧因子）。sha 抓不到"代码不同但算出来一样"的克隆。
        if metrics.get('signature') is not None:
            clones = similarity.clones_in_cache(metrics['signature'], similarity.CLONE_CORR)
            if clones:
                new_sharpe = metrics['sharpe'] if metrics['sharpe'] is not None else float('-inf')
                clone_sharpes = {}
                for cn, _ in clones:
                    rec = db.get_factor(cn)
                    clone_sharpes[cn] = rec['train_sharpe'] if rec and rec['train_sharpe'] is not None else float('-inf')
                best_old = max(clone_sharpes, key=clone_sharpes.get)
                if new_sharpe <= clone_sharpes[best_old]:
                    print(f'  - {name}: 截面近克隆 {best_old}（corr={clones[0][1]:.4f}）且夏普不更高'
                          f'（{new_sharpe:.3f} <= {clone_sharpes[best_old]:.3f}），拒绝')
                    results.append({'name': name, 'status': 'rejected',
                                    'reason': f'clone:{best_old}', 'sharpe': metrics['sharpe']})
                    continue
                victims = list(clone_sharpes)
                dedup_library.remove_factors(victims)
                print(f'  ~ {name}: 新高覆盖旧克隆 {victims}'
                      f'（新夏普={new_sharpe:.3f} > 旧最高={clone_sharpes[best_old]:.3f}）')

        thesis = _parse_thesis(code)
        parent_ids = json.dumps([p['name'] for p in job['parents']], ensure_ascii=False)
        dest = FACTORS_DIR / f'{name}.py'
        shutil.copyfile(path, dest)
        fid = db.add_factor(
            name=name, file_path=dest.relative_to(REPO_ROOT).as_posix(),
            code_sha256=sha, op=job['op'], generation=gen, params_count=n_params, status='passed',
            parent_ids=parent_ids, bt_start=cfg.start, bt_end=cfg.end, stock_pool=cfg.pool_label,
            train_sharpe=metrics['sharpe'], annualized=metrics['annualized'],
            max_dd=metrics['max_dd'], n_trades=metrics['n_trades'],
            thesis=thesis, run_id=cfg.run_id,
        )
        records.add_run(
            name, bt_start=cfg.start, bt_end=cfg.end, buy_n=cfg.buy_n,
            stock_pool=cfg.pool_label, run_id=cfg.run_id,
            dates=metrics['dates'], daily_returns=metrics['daily_returns'], topn=metrics['topn'],
            sharpe=metrics['sharpe'], annualized=metrics['annualized'],
            max_dd=metrics['max_dd'], n_trades=metrics['n_trades'],
        )
        if metrics.get('signature') is not None:
            ns, nstk = metrics['sig_shape']
            similarity.add_to_cache(name, metrics['signature'], {
                'dim': similarity.DEFAULT_DIM, 'seed': similarity.DEFAULT_SEED,
                'start': cfg.start, 'end': cfg.end, 'n_days': ns, 'n_stocks': nstk,
                'pool': cfg.pool_label,
            })
        print(f'  + #{fid} {name}: sharpe={metrics["sharpe"]:.3f} 年化={metrics["annualized"]:.1f}%'
              f' | {job["op"]} | 思路: {thesis or "(未提供)"}')
        results.append({'name': name, 'status': 'passed', 'sharpe': metrics['sharpe'],
                        'fid': fid, 'thesis': thesis, 'op': job['op'],
                        'parents': [p['name'] for p in job['parents']], 'generation': gen})

    passed = [r for r in results if r['status'] == 'passed']
    if passed:
        best = max(passed, key=lambda r: r['sharpe'])
        print(f'[gen {gen}] ★ 本代最优: {best["name"]} sharpe={best["sharpe"]:.3f}')
    return results


def evolve(cfg: RunConfig) -> list[dict]:
    rng = random.Random(cfg.seed)
    dates, stocks = evaluator.build_universe(cfg.start, cfg.end, cfg.pool_prefixes)
    print(f'回测口径: {cfg.start}~{cfg.end}, 股票池={cfg.pool_label}({len(stocks)}只), top{cfg.buy_n}')
    print(f'population={cfg.population} (父代{cfg.n_parents}/子代{cfg.n_offspring}), '
          f'代数={cfg.generations}, 模型={cfg.model}, 并发={cfg.concurrency}')

    # 整轮只加载一次 NPZ 面板，所有代/所有子代的连续性检查与因子计算共用（避免每个因子重复加载 580MB）
    panel = load_runtime_npz(dates, max_lookback=_PRELOAD_LOOKBACK)
    if panel is None:
        raise FileNotFoundError('runtime npz 未覆盖回测区间，无法预加载面板')
    print(f'已预加载 NPZ 面板（复用于全程）: {len(panel["trade_dates"])}d x {len(panel["stock_codes"])}s')

    elite_name = selection.best_factor_name(cfg.param_cap)
    all_results = []
    for gen in range(1, cfg.generations + 1):
        results = run_generation(gen, cfg, dates, stocks, rng, elite_name, panel=panel)
        all_results.extend(results)
        passed = [r for r in results if r['status'] == 'passed']
        if passed:
            elite_name = max(passed, key=lambda r: r['sharpe'])['name']  # 上一轮最优保留
    return all_results
