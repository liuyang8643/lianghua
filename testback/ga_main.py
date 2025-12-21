import os

os.environ['LOKY_PICKLER'] = 'pickle'  # 使用更快的pickle

import sys
import json
import random
import argparse
import time
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple
from joblib import Parallel, delayed, parallel_backend

from testback.logger import testback_logger
from testback.ga import NSGA2GeneticAlgorithm, load_progress_info, load_population_genes, save_progress, evaluate_individual_worker
from testback.account import StockAccountMocker
from core import init_stock_detail_cache
from core.database import allow_buy_stock_code_list, init_full_data
from core.strategies import TopN
from core.strategies._weights import FactorWeights, RANK_N
from utils.stock.time import get_trading_date_span
from utils.shared_memory import SharedMemoryCache

# 全局缓存
ga_cache = SharedMemoryCache('testback_cache', compress_level=6)


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='GA优化 - 多目标遗传算法')
    parser.add_argument('--population', type=int, default=24, help='种群大小（默认24）')
    parser.add_argument('--generations', type=int, default=100, help='进化代数（默认100）')
    parser.add_argument('--backtest-days', type=int, default=180, help='回测天数（默认180）')
    parser.add_argument('--initial-capital', type=float, default=500000.0, help='初始资金（默认500000）')
    parser.add_argument('--rank-n', type=int, default=RANK_N, help=f'选股数量（默认{RANK_N}）')
    parser.add_argument('--stock-limit', type=int, default=None, help='股票池数量限制（None=全部）')
    parser.add_argument('--resume-file', type=str, default=None, help='恢复文件路径')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = str(Path(args.resume_file).parent) if args.resume_file else f"results/ga_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    testback_logger.info(f'=' * 30+f"cpu数量:{os.cpu_count()}")
    worker_count = min(os.cpu_count(), 100)

    testback_logger.info(f'=' * 60)
    testback_logger.info(f'GA优化 - 多目标遗传算法')
    testback_logger.info(f'种群: {args.population}, 代数: {args.generations}, 进程: {worker_count}')
    testback_logger.info(f'回测: {args.backtest_days}天, 资金: {args.initial_capital:.0f}, 选股: {args.rank_n}')
    testback_logger.info(f'=' * 60)

    # 因子范围
    factor_ranges = {factor: (-1.0, 1.0) for factor in FactorWeights.keys()}

    # 恢复模式
    initial_population_genes = None
    start_generation = 0
    inherited_progress = None
    if args.resume_file:
        progress_info = load_progress_info(args.resume_file)
        start_generation = progress_info['current_generation'] + 1
        inherited_progress = progress_info['all_progress']
        initial_population_genes = load_population_genes(args.resume_file, args.population, factor_ranges)
        testback_logger.info(f"✅ 恢复: 第{start_generation}代, {len(initial_population_genes)}个体")

    # 获取股票池
    all_stocks = allow_buy_stock_code_list()
    if args.stock_limit:
        all_stocks = all_stocks[:args.stock_limit]
    testback_logger.info(f"股票池: {len(all_stocks)}只")

    # 预加载数据
    init_stock_detail_cache(all_stocks)
    init_full_data(all_stocks, '1d')

    # 准备数据集
    cache_keys = {}
    for name, start, end, is_random in [
        ('train', date(2014, 1, 1), date(2024, 1, 1), True),
        ('val', date(2024, 1, 1), date(2025, 1, 1), False),
        ('test', date(2025, 1, 1), date.today(), False)
    ]:
        dates = get_trading_date_span(start, end)
        if is_random:
            dates = sorted(random.sample(dates, min(args.backtest_days, len(dates))))
        testback_logger.info(f"📊 {name}: {dates[0].strftime('%Y-%m-%d')}~{dates[-1].strftime('%Y-%m-%d')} ({len(dates)}天)")

        # 使用多进程（与 main.py 保持一致）
        with parallel_backend('loky', n_jobs=worker_count, inner_max_num_threads=1):
            topNs = Parallel(
                n_jobs=worker_count,
                prefer='processes',
                batch_size=1,
            )(
                delayed(TopN)(all_stocks, datetime.combine(d, datetime.min.time()))
                for d in dates
            )

        cache_key = f"topn_{name}_{timestamp}"
        ga_cache.put(cache_key, topNs)
        cache_keys[name] = cache_key

    # 定义每代回调：保存进度 + 评估验证/测试集
    def _generation_callback(gen, pareto_front, ga_instance):
        """每代结束后：保存进度 + 评估Pareto前沿在验证/测试集的表现"""
        # 1. 保存训练集进度
        save_progress(gen, ga_instance, result_dir, inherited_progress=inherited_progress)

        # 2. 评估Pareto前沿的top个体（最多3个）
        top_n = min(3, len(pareto_front))
        if top_n == 0:
            return

        testback_logger.info(f"\n{'='*60}")
        testback_logger.info(f"[第{gen}代] Pareto前沿Top{top_n}在三个数据集的表现:")

        # 收集验证/测试集结果
        val_test_results = []

        for idx in range(top_n):
            ind = pareto_front[idx]
            train_excess, train_sharpe, train_calmar = ind.fitness.values

            testback_logger.info(f"\n  [{idx}] 训练集: 超额={train_excess:+.4f} 夏普={train_sharpe:.2f} Calmar={train_calmar:.2f}")

            # 验证集评估
            val_excess, val_sharpe, val_calmar, val_metrics = evaluate_individual_worker(
                genes=ind.genes,
                topn_cache_key=cache_keys['val'],
                initial_cash=args.initial_capital,
                rank_n=args.rank_n,
                commission_rate=0.0003,
                min_commission=5.0,
                individual_idx=idx,
                generation=gen
            )
            testback_logger.info(f"      验证集: 超额={val_excess:+.4f} 夏普={val_sharpe:.2f} Calmar={val_calmar:.2f}")

            # 测试集评估
            test_excess, test_sharpe, test_calmar, test_metrics = evaluate_individual_worker(
                genes=ind.genes,
                topn_cache_key=cache_keys['test'],
                initial_cash=args.initial_capital,
                rank_n=args.rank_n,
                commission_rate=0.0003,
                min_commission=5.0,
                individual_idx=idx,
                generation=gen
            )
            testback_logger.info(f"      测试集: 超额={test_excess:+.4f} 夏普={test_sharpe:.2f} Calmar={test_calmar:.2f}")

            # 收集结果
            val_test_results.append({
                'individual_idx': idx,
                'genes': ind.genes,
                'train': {
                    'excess': train_excess,
                    'sharpe': train_sharpe,
                    'calmar': train_calmar
                },
                'val': {
                    'excess': val_excess,
                    'sharpe': val_sharpe,
                    'calmar': val_calmar,
                    'metrics': val_metrics
                },
                'test': {
                    'excess': test_excess,
                    'sharpe': test_sharpe,
                    'calmar': test_calmar,
                    'metrics': test_metrics
                }
            })

        testback_logger.info(f"{'='*60}")

        # 3. 将验证/测试集结果追加到 progress.json
        progress_file = os.path.join(result_dir, 'progress.json')
        if os.path.exists(progress_file):
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)

            # 找到当前代的数据，添加验证/测试集结果
            for gen_data in progress_data['generations']:
                if gen_data['generation'] == gen:
                    gen_data['val_test_evaluation'] = val_test_results
                    break

            # 保存回文件
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)

    # 创建并运行GA
    testback_logger.info("开始GA优化...")
    ga = NSGA2GeneticAlgorithm(
        factor_ranges=factor_ranges,
        fitness_function=evaluate_individual_worker,
        fitness_params={
            'topn_cache_key': cache_keys['train'],
            'initial_cash': args.initial_capital,
            'rank_n': args.rank_n,
            'commission_rate': 0.0003,
            'min_commission': 5.0
        },
        population_size=args.population,
        mutation_rate=0.2,
        crossover_rate=0.8,
        max_generations=args.generations,
        max_workers=worker_count,
        progress_callback=_generation_callback,
        initial_genes=None,
        initial_population_genes=initial_population_genes,
        start_generation=start_generation,
        result_dir=result_dir,
        elite_pool_size=args.population
    )

    try:
        population, pareto_front = ga.run()
    except KeyboardInterrupt:
        testback_logger.warning("⚠️ 用户中断")
        sys.exit(0)

    testback_logger.info("GA优化完成")

    # 评估 Pareto 前沿（前3个）
    testback_logger.info("\n评估 Pareto 前沿...")
    pareto_evaluation = []
    for idx, ind in enumerate(pareto_front[:3]):
        testback_logger.info(f"\n[{idx}] 权重: {ind.genes}")

        eval_results = {}
        for dataset_name in ['train', 'val', 'test']:
            if dataset_name == 'train':
                result = {
                    'excess': ind.fitness.values[0],
                    'sharpe': ind.fitness.values[1],
                    'calmar': ind.fitness.values[2],
                    'metrics': ind.detailed_metrics
                }
            else:
                excess, sharpe, calmar, metrics = evaluate_individual_worker(
                    genes=ind.genes,
                    topn_cache_key=cache_keys[dataset_name],
                    initial_cash=args.initial_capital,
                    rank_n=args.rank_n,
                    commission_rate=0.0003,
                    min_commission=5.0,
                    individual_idx=0,
                    generation=999
                )
                result = {'excess': excess, 'sharpe': sharpe, 'calmar': calmar, 'metrics': metrics}

            eval_results[dataset_name] = result
            testback_logger.info(
                f"  {dataset_name}: 超额={result['excess']:+.4f} 夏普={result['sharpe']:.2f} "
                f"Calmar={result['calmar']:.2f} 年化={result['metrics']['annualized_return']:.4f}"
            )

        pareto_evaluation.append({'genes': ind.genes, **eval_results})

    # 保存结果
    from utils.stock.info import baseline_stock_code
    final_result = {
        'timestamp': timestamp,
        'mode': 'multi_objective',
        'config': {
            'population': args.population,
            'generations': args.generations,
            'backtest_days': args.backtest_days,
            'initial_capital': args.initial_capital,
            'rank_n': args.rank_n,
            'max_workers': worker_count,
            'stock_limit': args.stock_limit,
            'baseline': baseline_stock_code
        },
        'pareto_front': [
            {
                'genes': ind.genes,
                'objective1_excess': ind.fitness.values[0],
                'objective2_sharpe': ind.fitness.values[1],
                'objective3_calmar': ind.fitness.values[2],
                **ind.detailed_metrics
            } for ind in pareto_front
        ],
        'pareto_evaluation_top3': pareto_evaluation,
        'total_time_seconds': sum(ga.generation_time_history) if ga.generation_time_history else 0
    }

    with open(os.path.join(result_dir, 'final_result.json'), 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)

    # 输出摘要
    testback_logger.info(f"\n📊 Pareto前沿: {len(pareto_front)}个体")
    testback_logger.info("="*60)
    for idx, ind in enumerate(pareto_front[:10]):
        testback_logger.info(f"  [{idx}] 超额={ind.fitness.values[0]:+.4f} 夏普={ind.fitness.values[1]:.2f} Calmar={ind.fitness.values[2]:.2f}")
    testback_logger.info("="*60)

    if ga.generation_time_history:
        total_time = sum(ga.generation_time_history)
        testback_logger.info(f"总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟), 平均: {total_time/len(ga.generation_time_history):.1f}秒/代")

    testback_logger.info("✅ 完成")
