"""分析所有GA历史数据中各因子的权重均值、方差、正负权重夏普期望"""
import os, json, pickle, sys
from collections import defaultdict
import numpy as np

RESULTS_DIR = 'results'

def main():
    # 收集所有 ga_* 目录
    ga_dirs = [d for d in os.listdir(RESULTS_DIR)
               if d.startswith('ga_') and os.path.isdir(os.path.join(RESULTS_DIR, d))]
    print(f'找到 {len(ga_dirs)} 个GA运行目录')

    # 聚合数据结构
    # 1. 权重统计 (from generation_results.pkl best_weights)
    factor_weights = defaultdict(list)       # factor -> [weight, ...]
    factor_sharpes = defaultdict(list)       # factor -> [(weight, sharpe), ...]
    pos_sharpes = defaultdict(list)          # factor -> [sharpe when weight>0]
    neg_sharpes = defaultdict(list)          # factor -> [sharpe when weight<0]

    gen_count = 0
    indiv_count = 0

    for i, d in enumerate(ga_dirs):
        dpath = os.path.join(RESULTS_DIR, d)

        # 优先用 all_results.jsonl (包含所有个体)
        jsonl_path = os.path.join(dpath, 'all_results.jsonl')
        pkl_path = os.path.join(dpath, 'generation_results.pkl')

        if os.path.exists(jsonl_path):
            # 流式读取 JSONL
            with open(jsonl_path, 'r') as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    config = r.get('config', {})
                    weights = config.get('weights', {})
                    sharpe = r.get('sharpe', 0)
                    if not weights:
                        continue
                    indiv_count += 1
                    for fname, w in weights.items():
                        factor_weights[fname].append(w)
                        factor_sharpes[fname].append((w, sharpe))
                        if w > 0:
                            pos_sharpes[fname].append(sharpe)
                        elif w < 0:
                            neg_sharpes[fname].append(sharpe)
        elif os.path.exists(pkl_path):
            # 回退到 pkl best_weights
            with open(pkl_path, 'rb') as f:
                gen_data = pickle.load(f)
            for g in gen_data:
                gen_count += 1
                weights = g.get('best_weights', {})
                sharpe = g.get('max_fitness', 0)
                for fname, w in weights.items():
                    factor_weights[fname].append(w)
                    factor_sharpes[fname].append((w, sharpe))
                    if w > 0:
                        pos_sharpes[fname].append(sharpe)
                    elif w < 0:
                        neg_sharpes[fname].append(sharpe)

        if (i + 1) % 20 == 0:
            print(f'  已处理 {i+1}/{len(ga_dirs)} 目录, {indiv_count} 个体, {gen_count} 代')

    print(f'\n总计: {indiv_count} 个体, {gen_count} 代 (pkl回退)')
    print(f'涉及因子: {sorted(factor_weights.keys())}\n')

    # 计算统计
    factor_names = sorted(factor_weights.keys())
    print(f'{"因子":24s} {"权重均值":>8s} {"权重方差":>8s} {"权重std":>8s}  '
          f'{"正权夏普均值":>12s} {"正权样本":>8s}  '
          f'{"负权夏普均值":>12s} {"负权样本":>8s}  '
          f'{"夏普差":>8s} {"净方向":>6s}')
    print('-' * 130)

    for fname in factor_names:
        w = np.array(factor_weights[fname])
        w_mean = np.mean(w)
        w_var = np.var(w)
        w_std = np.std(w)

        pos_s = pos_sharpes.get(fname, [])
        neg_s = neg_sharpes.get(fname, [])
        pos_mean = np.mean(pos_s) if pos_s else float('nan')
        neg_mean = np.mean(neg_s) if neg_s else float('nan')
        pos_n = len(pos_s)
        neg_n = len(neg_s)
        sharpe_diff = pos_mean - neg_mean

        # 净方向: 正权重夏普 > 负权重夏普 => 正向因子
        if pos_n > 50 and neg_n > 50:
            if sharpe_diff > 0.05:
                direction = '正向'
            elif sharpe_diff < -0.05:
                direction = '负向'
            else:
                direction = '中性'
        else:
            direction = '样本不足'

        print(f'{fname:24s} {w_mean:8.4f} {w_var:8.4f} {w_std:8.4f}  '
              f'{pos_mean:12.4f} {pos_n:8d}  '
              f'{neg_mean:12.4f} {neg_n:8d}  '
              f'{sharpe_diff:8.4f} {direction:>6s}')

    # 详细: 权重分布直方图
    print('\n\n=== 权重分布详情 ===')
    for fname in factor_names:
        w = np.array(factor_weights[fname])
        pos_ratio = np.sum(w > 0) / len(w) * 100
        neg_ratio = np.sum(w < 0) / len(w) * 100
        zero_ratio = np.sum(w == 0) / len(w) * 100
        print(f'{fname:24s}  n={len(w):7d}  正权:{pos_ratio:5.1f}%  负权:{neg_ratio:5.1f}%  零权:{zero_ratio:5.1f}%  '
              f'中位数:{np.median(w):6.3f}  P25:{np.percentile(w,25):6.3f}  P75:{np.percentile(w,75):6.3f}')

    # 按夏普差排序
    print('\n\n=== 因子净贡献排序 (正权夏普 - 负权夏普) ===')
    ranked = []
    for fname in factor_names:
        pos_s = pos_sharpes.get(fname, [])
        neg_s = neg_sharpes.get(fname, [])
        if len(pos_s) > 50 and len(neg_s) > 50:
            ranked.append((fname, np.mean(pos_s) - np.mean(neg_s), np.mean(pos_s), np.mean(neg_s)))
    ranked.sort(key=lambda x: x[1], reverse=True)
    for i, (fname, diff, pm, nm) in enumerate(ranked):
        bar = '█' * max(0, int(diff * 50)) if diff > 0 else '▓' * max(0, int(-diff * 50))
        print(f'{i+1:2d}. {fname:24s} Δ={diff:+.4f}  正权夏普={pm:.4f}  负权夏普={nm:.4f}  {bar}')


if __name__ == '__main__':
    main()
