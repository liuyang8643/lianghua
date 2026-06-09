"""
GA 收敛速度实验 — 在合成景观上反复运行 GA，统计收敛曲线。

用法:
  uv run python -m testback.run_ga_convergence --trials 100 --gens 80 --kind rastrigin --profile core5
  uv run python -m testback.run_ga_convergence --trials 100 --gens 80 --kind nk_chaotic --profile core5

输出 results/ga_convergence/:
  - aggregate_*.json      每代平均收敛指标 (±std)
  - detail_*.jsonl        逐试验逐代明细
"""

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from core.ga import (
    build_individual_config,
    generate_initial_configs,
    get_profile,
    get_profile_search_spaces,
    get_profile_weight_search_spaces,
    sample_position_count, sample_factor_choice, sample_stock_pool,
    sample_holding_period, sample_timing_base, sample_timing_leverage,
    sample_timing_direction, sample_timing_window, sample_timing_index,
)
from testback.run_ga import ga_optimizer, _config_key
from testback.ga_landscape import (
    GALandscape,
    _ConfigEncoder,
    compute_convergence_metrics,
)


def run_ga_on_landscape(
    landscape: GALandscape,
    profile_name: str,
    population_size: int = 24,
    n_generations: int = 80,
    seed: int = 42,
) -> list[dict]:
    """在合成景观上运行完整 GA，返回每代收敛指标。"""
    random.seed(seed)
    np.random.seed(seed)

    encoder = landscape.encoder

    ga_state = {'population': [], 'hall_of_fame': [], 'fitness_cache': {}}
    ga_cache = {}

    gen_metrics = []

    for gen in range(n_generations):
        if gen == 0:
            configs = generate_initial_configs(population_size, profile_name=profile_name)
            results = []
            for cfg in configs:
                r = landscape.evaluate(cfg)
                r['individual_config'] = cfg
                results.append(r)
                ga_cache[_config_key(cfg)] = r
        else:
            results = [{'individual_config': cfg,
                        'sharpe': ga_cache.get(_config_key(cfg), {}).get('sharpe', -999)}
                       for cfg in ga_state['population']]
            configs = ga_optimizer(results, ga_state, population_size=population_size,
                                   profile_name=profile_name, ga_cache=ga_cache, gen=gen)
            results = []
            for cfg in configs:
                r = landscape.evaluate(cfg)
                r['individual_config'] = cfg
                results.append(r)
                ga_cache[_config_key(cfg)] = r

        ga_state['population'] = configs

        metrics = compute_convergence_metrics(configs, encoder)
        metrics['generation'] = gen
        metrics['best_sharpe'] = max(r['sharpe'] for r in results)
        metrics['mean_sharpe'] = float(np.mean([r['sharpe'] for r in results]))
        gen_metrics.append(metrics)

    return gen_metrics


def main():
    parser = argparse.ArgumentParser(description='GA收敛速度实验')
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--gens', type=int, default=80)
    parser.add_argument('--pop', type=int, default=24)
    parser.add_argument('--kind', type=str, default='rastrigin',
                        choices=['rastrigin', 'ackley', 'griewank', 'sphere',
                                 'nk', 'nk_chaotic', 'deceptive'])
    parser.add_argument('--profile', type=str, default='core5')
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--dim', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    profile_name = args.profile
    landscape = GALandscape(kind=args.kind, dim=args.dim, profile_name=profile_name,
                            seed=args.seed, noise=args.noise)

    stats = landscape.landscape_stats()
    print("=" * 60)
    print("景观特征:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  搜索空间维度: {landscape.encoder.dim if landscape.encoder else landscape.dim}")
    print("=" * 60)

    output_dir = Path('results/ga_convergence')
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_tag = f"{args.kind}_n{args.noise}_p{args.pop}_g{args.gens}_t{args.trials}_{timestamp}"

    trials_detail = []
    all_gen_metrics = {}

    t0 = time.time()
    for trial in range(args.trials):
        seed = args.seed + trial
        gen_metrics = run_ga_on_landscape(
            landscape, profile_name,
            population_size=args.pop,
            n_generations=args.gens,
            seed=seed,
        )
        for gm in gen_metrics:
            gm['trial'] = trial
        trials_detail.extend(gen_metrics)

        for gm in gen_metrics:
            g = gm['generation']
            if g not in all_gen_metrics:
                all_gen_metrics[g] = {}
            for k, v in gm.items():
                if k in ('generation', 'trial'):
                    continue
                all_gen_metrics[g].setdefault(k, []).append(v)

        if (trial + 1) % max(1, args.trials // 10) == 0:
            elapsed = time.time() - t0
            remaining = (elapsed / (trial + 1)) * (args.trials - trial - 1)
            print(f"  [{trial+1}/{args.trials}] {elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining")

    total_elapsed = time.time() - t0
    print(f"\n总耗时: {total_elapsed:.1f}s ({total_elapsed/args.trials:.3f}s/trial)")

    # 聚合
    agg = {}
    for gen_idx in sorted(all_gen_metrics.keys()):
        agg[f'gen_{gen_idx}'] = {
            k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
            for k, v in all_gen_metrics[gen_idx].items()
        }

    agg_file = output_dir / f'aggregate_{run_tag}.json'
    agg_file.write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding='utf-8')

    detail_file = output_dir / f'detail_{run_tag}.jsonl'
    with open(detail_file, 'w', encoding='utf-8') as f:
        for d in trials_detail:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    print(f"结果: {agg_file}")
    print(f"明细: {detail_file}")

    # 摘要
    print("\n" + "=" * 60)
    print(f"{'指标':<30} {'Gen 1':>12} {'Gen 10':>12} {'Gen 50':>12} {'Gen '+str(args.gens-1):>12}")
    print("-" * 60)
    for metric in ['uniqueness', 'genotypic_variance', 'best_sharpe']:
        row = f"{metric:<30}"
        for cg in [1, 10, min(50, args.gens - 1), args.gens - 1]:
            gk = f'gen_{cg}'
            if gk in agg and metric in agg[gk]:
                m = agg[gk][metric]
                row += f" {m['mean']:>8.3f}±{m['std']:.2f}"
            else:
                row += " " * 13
        print(row)

    # 收敛代数
    conv_gens = []
    for trial in range(args.trials):
        trial_data = [d for d in trials_detail if d['trial'] == trial]
        for d in trial_data:
            if d.get('uniqueness', 1.0) < 0.3:
                conv_gens.append(d['generation'])
                break
        else:
            conv_gens.append(args.gens)
    print(f"\n收敛代数 (uniqueness < 0.3): mean={np.mean(conv_gens):.1f} ± {np.std(conv_gens):.1f}")


if __name__ == '__main__':
    main()
