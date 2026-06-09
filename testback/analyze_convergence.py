"""
收敛实验分析 — 读取 aggregate JSON，生成对比报告。
"""
import json
import sys
from pathlib import Path

import numpy as np


def load_aggregate(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze(agg: dict) -> dict:
    """从聚合数据提取关键收敛指标"""
    gens = sorted(int(k.replace('gen_', '')) for k in agg)
    n_gens = len(gens)

    # 最佳 sharpe 随代数变化
    best_sharpes = [agg[f'gen_{g}']['best_sharpe']['mean'] for g in gens]

    # 找到首次达到 90% / 95% 最终最优的代数
    final_best = best_sharpes[-1]
    gen_90 = next((g for g in gens if agg[f'gen_{g}']['best_sharpe']['mean'] >= 0.90 * final_best), gens[-1])
    gen_95 = next((g for g in gens if agg[f'gen_{g}']['best_sharpe']['mean'] >= 0.95 * final_best), gens[-1])

    # uniqueness 衰减：首代 vs 末代
    uniq_first = agg[f'gen_{gens[0]}']['uniqueness']['mean']
    uniq_last = agg[f'gen_{gens[-1]}']['uniqueness']['mean']

    # genotypic_variance 变化率
    gv_first = agg[f'gen_{gens[0]}']['genotypic_variance']['mean']
    gv_last = agg[f'gen_{gens[-1]}']['genotypic_variance']['mean']
    gv_change = (gv_last - gv_first) / gv_first if gv_first else 0

    # 平台期检测：最后 20 代的 best_sharpe 改善量
    late_gens = gens[-20:]
    early_late = agg[f'gen_{late_gens[0]}']['best_sharpe']['mean']
    improvement_rate = (final_best - early_late) / (abs(early_late) or 1)

    return {
        'final_best_sharpe': round(final_best, 4),
        'gen_to_90pct': gen_90,
        'gen_to_95pct': gen_95,
        'uniqueness_start': round(uniq_first, 3),
        'uniqueness_end': round(uniq_last, 3),
        'gv_change_pct': round(gv_change * 100, 1),
        'late_improvement_pct': round(improvement_rate * 100, 2),
        'n_gens': n_gens,
    }


def main():
    results_dir = Path('results/ga_convergence')
    if not results_dir.is_dir():
        print("没有找到结果目录 results/ga_convergence")
        sys.exit(1)

    # 找最新的 aggregate 文件
    agg_files = sorted(results_dir.glob('aggregate_*.json'), key=lambda p: p.stat().st_mtime, reverse=True)

    # 按景观类型分组
    by_kind = {}
    for f in agg_files:
        # 解析文件名: aggregate_{kind}_n{noise}_p{pop}_g{gens}_t{trials}_{timestamp}.json
        name = f.stem.replace('aggregate_', '')
        parts = name.split('_')
        kind = parts[0]
        if kind not in by_kind or f.stat().st_mtime > by_kind[kind][0].stat().st_mtime:
            if kind not in by_kind:
                by_kind[kind] = (f,)
            else:
                by_kind[kind] = (f,)

    # 重新按最新修改时间整理
    by_kind = {}
    for f in agg_files:
        name = f.stem.replace('aggregate_', '')
        kind = name.split('_')[0]
        if kind not in by_kind or f.stat().st_mtime > by_kind[kind][0].stat().st_mtime:
            by_kind[kind] = [f]
        elif f.stat().st_mtime == by_kind[kind][0].stat().st_mtime:
            by_kind[kind].append(f)

    # 去重，每类只取最新的
    final = {}
    for k, files in by_kind.items():
        final[k] = max(files, key=lambda f: f.stat().st_mtime)

    if not final:
        print("没有找到结果文件")
        return

    print("=" * 90)
    print("GA 收敛对比分析 (各景观 100 trials × 100 gens)")
    print("=" * 90)
    print(f"{'景观':<16} {'最终Sharpe':>10} {'90%代数':>8} {'95%代数':>8} "
          f"{'Uniq始':>8} {'Uniq末':>8} {'GV变化%':>8} {'末段改善%':>10}")
    print("-" * 90)

    results = {}
    for kind, f in sorted(final.items()):
        agg = load_aggregate(f)
        r = analyze(agg)
        results[kind] = r
        print(f"{kind:<16} {r['final_best_sharpe']:>10.4f} {r['gen_to_90pct']:>8} {r['gen_to_95pct']:>8} "
              f"{r['uniqueness_start']:>8.3f} {r['uniqueness_end']:>8.3f} "
              f"{r['gv_change_pct']:>8.1f} {r['late_improvement_pct']:>10.2f}")

    print("-" * 90)
    print()

    # 诊断结论
    print("=" * 90)
    print("诊断结论")
    print("=" * 90)

    for kind, r in sorted(results.items()):
        issues = []

        # 收敛过快
        if r['gen_to_95pct'] < 10:
            issues.append(f"收敛极快 (95%在gen {r['gen_to_95pct']})，可能 selection pressure 过高")
        elif r['gen_to_95pct'] > 80:
            issues.append(f"收敛极慢 (95%在gen {r['gen_to_95pct']})，探索有余但开发不足")

        # 末段改善
        if r['late_improvement_pct'] < 0.5:
            issues.append(f"末段几乎无改善 ({r['late_improvement_pct']:.2f}%)，已陷入局部最优平台")
        elif r['late_improvement_pct'] > 5:
            issues.append(f"末段仍在明显改善 ({r['late_improvement_pct']:.2f}%)，可能尚未收敛")

        # 多样性
        if r['uniqueness_end'] < 0.3:
            issues.append(f"种群多样性崩溃 ({r['uniqueness_end']:.1%})，早熟收敛风险高")
        elif r['uniqueness_end'] > 0.9:
            issues.append(f"种群多样性极高 ({r['uniqueness_end']:.1%})，搜索偏随机游走")

        if not issues:
            issues.append("收敛行为健康，探索/开发平衡良好")

        print(f"\n[{kind}]")
        for issue in issues:
            print(f"  • {issue}")

    # 汇总建议
    print(f"\n{'─' * 60}")
    print("综合建议:")
    avg_conv = sum(r['gen_to_95pct'] for r in results.values()) / len(results)
    avg_late = sum(r['late_improvement_pct'] for r in results.values()) / len(results)

    if avg_conv < 20 and avg_late < 0.5:
        print("  → GA 趋向于快速早熟收敛。在真实回测中，可能过度拟合训练期噪声。")
        print("    建议: 增大 immigrant_frac (0.15→0.25)，减小 tournament_k (3→2)，增大变异率")
    elif avg_conv > 60:
        print("  → GA 收敛过慢，搜索效率偏低。100代后仍在探索。")
        print("    建议: 增大 tournament_k (3→5)，减小 immigrant_frac (0.15→0.05)，或增大种群")
    else:
        print("  → GA 收敛速度适中，表现健康。")


if __name__ == '__main__':
    main()
