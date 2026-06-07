"""LLM-GA 因子进化 CLI 入口。

示例:
  uv run python run_llm_ga.py --generations 3 --per-gen 2 --start 19930101 --end 20181231
"""
import argparse

from factor_db import db
from llm_ga.config import RunConfig, preset_core5, preset_core9, preset_core14
from llm_ga.loop import evolve

_POOL_PRESETS = {
    'all_A': ('60', '00', '30', '688'),
    'main': ('60', '00'),
    'gem': ('30',),
    'star': ('688',),
}

_CORE_PRESETS = {
    'core5': preset_core5,
    'core9': preset_core9,
    'core14': preset_core14,
}


def _ensure_seeds():
    if not db.list_factors():
        from factor_db import seed_registry
        seed_registry.main()


def main():
    p = argparse.ArgumentParser(description='LLM-GA 因子进化（MadEvolve 式）')
    p.add_argument('--generations', type=int, default=100)
    p.add_argument('--population', type=int, default=10, help='父代+子代总数，默认 10 → 5 父代 + 5 子代')
    p.add_argument('--start', type=str, default='20100101')
    p.add_argument('--end', type=str, default='20260605')
    p.add_argument('--buy-n', type=int, default=20)
    p.add_argument('--pool', type=str, default='all_A', choices=list(_POOL_PRESETS))
    p.add_argument('--param-cap', type=int, default=20)
    p.add_argument('--crossover-ratio', type=float, default=0.3)
    p.add_argument('--concurrency', type=int, default=5, help='同一代内并发产因子的子进程数')
    p.add_argument('--model', type=str, default='deepseek-v4-pro', help='产因子模型')
    p.add_argument('--verify-model', type=str, default='deepseek-v4-flash', help='verify 红线审查模型')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--preset', type=str, default=None, choices=list(_CORE_PRESETS),
                   help='限定因子池预设：core5/core9/core14')
    p.add_argument('--no-llm-verify', action='store_true', help='关闭 claude -p 的 LLM verify 硬闸门')
    args = p.parse_args()

    _ensure_seeds()

    if args.preset:
        cfg = _CORE_PRESETS[args.preset](
            start=args.start, end=args.end, buy_n=args.buy_n,
            pool_prefixes=_POOL_PRESETS[args.pool], pool_label=args.pool,
            generations=args.generations,
            param_cap=args.param_cap, crossover_ratio=args.crossover_ratio,
            concurrency=args.concurrency, model=args.model, verify_model=args.verify_model,
            seed=args.seed, llm_verify=not args.no_llm_verify,
        )
    else:
        n_offspring = args.population // 2
        cfg = RunConfig(
            start=args.start, end=args.end, buy_n=args.buy_n,
            pool_prefixes=_POOL_PRESETS[args.pool], pool_label=args.pool,
            generations=args.generations, population=args.population,
            n_offspring=n_offspring, n_parents=args.population - n_offspring,
            n_parents_crossover=min(5, args.population - n_offspring),
            param_cap=args.param_cap, crossover_ratio=args.crossover_ratio,
            concurrency=args.concurrency, model=args.model, verify_model=args.verify_model,
            seed=args.seed, llm_verify=not args.no_llm_verify,
        )
    results = evolve(cfg)

    passed = [r for r in results if r['status'] == 'passed']
    print(f'\n完成: {len(passed)}/{len(results)} 通过')
    for r in sorted(passed, key=lambda x: x['sharpe'], reverse=True):
        print(f"  {r['name']:<28} sharpe={r['sharpe']:.3f}")


if __name__ == '__main__':
    main()
