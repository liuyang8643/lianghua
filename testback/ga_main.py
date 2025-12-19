"""
GA优化主程序

用法：
    python -m testback.ga_main --population 24 --generations 100
    python -m testback.ga_main --resume-file results/ga_xxx/progress.json
    python -m testback.ga_main --resume-best-only --resume-file results/ga_xxx/final_result.json

参数：
    --population: 种群大小（默认24）
    --generations: 进化代数（默认100）
    --backtest-days: 回测天数（默认180天）
    --initial-capital: 初始资金（默认500000）
    --max-workers: 最大进程数（默认=population）
    --resume-file: 恢复文件路径
    --resume-best-only: 只恢复最佳个体

示例：
    # 新运行
    python -m testback.ga_main --population 24 --generations 100 --backtest-days 180

    # 恢复运行（继承整个种群）
    python -m testback.ga_main --resume-file results/ga_20250118_120000/progress.json

    # 恢复运行（只恢复最佳个体）
    python -m testback.ga_main --resume-best-only --resume-file results/ga_20250118_120000/final_result.json
"""

import sys
import os
import argparse
import json
from datetime import datetime, date, timedelta
from pathlib import Path

from testback.logger import testback_logger
from testback.ga import (
    NSGA2GeneticAlgorithm,
    evaluate_individual,
    load_previous_best,
    load_progress_info,
    load_population_genes,
    save_progress
)
from core.strategies._weights import FactorWeights, RANK_N
from core.database import allow_buy_stock_code_list
from utils.stock.time import get_trading_date_span
from utils.parallel import batch_run_threads
from utils.shared_memory import SharedMemoryCache


def prepare_data(backtest_days: int = 180, stock_limit: int = None, use_random_sample: bool = True):
    """准备回测数据（支持随机采样训练集 + 验证集/测试集）

    Args:
        backtest_days: 回测天数
        stock_limit: 股票数量限制（None=全部）
        use_random_sample: 是否从历史范围随机选择训练集

    Returns:
        dict: {
            'train': 训练集 cache_key,
            'val': 验证集 cache_key (2024-01-01 to 2025-01-01),
            'test': 测试集 cache_key (2025-01-01 to today)
        }
    """
    testback_logger.info("="*60)
    testback_logger.info("准备回测数据（训练集/验证集/测试集）...")
    testback_logger.info("="*60)

    # 获取股票池
    all_stocks = allow_buy_stock_code_list()
    if stock_limit:
        all_stocks = all_stocks[:stock_limit]
    testback_logger.info(f"股票池: {len(all_stocks)}只股票")

    from core.strategies import TopN
    topn_cache = SharedMemoryCache('testback_cache', compress_level=6)
    cache_keys = {}

    # === 1. 训练集：随机采样或最近 N 天 ===
    if use_random_sample:
        # 从 2022-2023 年范围随机选择 backtest_days 个交易日
        import random
        train_pool_start = date(2022, 1, 1)
        train_pool_end = date(2023, 12, 31)
        all_train_dates = get_trading_date_span(train_pool_start, train_pool_end)

        if len(all_train_dates) < backtest_days:
            testback_logger.warning(f"历史数据不足，使用全部 {len(all_train_dates)} 天")
            train_dates = all_train_dates
        else:
            train_dates = sorted(random.sample(all_train_dates, backtest_days))

        testback_logger.info(f"📊 训练集（随机采样）: {train_dates[0].strftime('%Y-%m-%d')} ~ {train_dates[-1].strftime('%Y-%m-%d')} ({len(train_dates)}个交易日)")
    else:
        # 使用最近 backtest_days 天
        today = date.today()
        start_date = today - timedelta(days=int(backtest_days * 1.4))
        train_dates = get_trading_date_span(start_date, today)[-backtest_days:]
        testback_logger.info(f"📊 训练集（最近）: {train_dates[0].strftime('%Y-%m-%d')} ~ {train_dates[-1].strftime('%Y-%m-%d')} ({len(train_dates)}个交易日)")

    # 计算训练集 TopN
    testback_logger.info("计算训练集因子得分（多线程）...")
    t_start = datetime.now()
    train_topNs = batch_run_threads(
        func=TopN,
        args_list=[[all_stocks, datetime.combine(d, datetime.min.time())] for d in train_dates],
        max_workers=8,
    )
    t_end = datetime.now()
    testback_logger.info(f"  耗时: {(t_end - t_start).total_seconds():.1f}秒")

    train_cache_key = f"topn_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    topn_cache.put(train_cache_key, train_topNs)
    cache_keys['train'] = train_cache_key

    # === 2. 验证集：2024-01-01 to 2025-01-01 ===
    val_start = date(2024, 1, 1)
    val_end = date(2025, 1, 1)
    val_dates = get_trading_date_span(val_start, val_end)
    testback_logger.info(f"📊 验证集: {val_dates[0].strftime('%Y-%m-%d')} ~ {val_dates[-1].strftime('%Y-%m-%d')} ({len(val_dates)}个交易日)")

    testback_logger.info("计算验证集因子得分（多线程）...")
    t_start = datetime.now()
    val_topNs = batch_run_threads(
        func=TopN,
        args_list=[[all_stocks, datetime.combine(d, datetime.min.time())] for d in val_dates],
        max_workers=8,
    )
    t_end = datetime.now()
    testback_logger.info(f"  耗时: {(t_end - t_start).total_seconds():.1f}秒")

    val_cache_key = f"topn_val_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    topn_cache.put(val_cache_key, val_topNs)
    cache_keys['val'] = val_cache_key

    # === 3. 测试集：2025-01-01 to today ===
    test_start = date(2025, 1, 1)
    test_end = date.today()
    test_dates = get_trading_date_span(test_start, test_end)
    testback_logger.info(f"📊 测试集: {test_dates[0].strftime('%Y-%m-%d')} ~ {test_dates[-1].strftime('%Y-%m-%d')} ({len(test_dates)}个交易日)")

    testback_logger.info("计算测试集因子得分（多线程）...")
    t_start = datetime.now()
    test_topNs = batch_run_threads(
        func=TopN,
        args_list=[[all_stocks, datetime.combine(d, datetime.min.time())] for d in test_dates],
        max_workers=8,
    )
    t_end = datetime.now()
    testback_logger.info(f"  耗时: {(t_end - t_start).total_seconds():.1f}秒")

    test_cache_key = f"topn_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    topn_cache.put(test_cache_key, test_topNs)
    cache_keys['test'] = test_cache_key

    testback_logger.info(f"✅ 数据准备完成")
    return cache_keys


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='GA优化 - 多目标遗传算法',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # 基本参数
    parser.add_argument('--population', type=int, default=24, help='种群大小（默认24）')
    parser.add_argument('--generations', type=int, default=100, help='进化代数（默认100）')
    parser.add_argument('--backtest-days', type=int, default=180, help='回测天数（默认180）')
    parser.add_argument('--initial-capital', type=float, default=500000.0, help='初始资金（默认500000）')
    parser.add_argument('--rank-n', type=int, default=RANK_N, help=f'选股数量（默认{RANK_N}）')
    parser.add_argument('--stock-limit', type=int, default=None, help='股票池数量限制（None=全部）')

    # 并行参数
    parser.add_argument('--max-workers', type=int, default=None, help='最大进程数（默认=population）')

    # 恢复参数
    parser.add_argument('--resume-file', type=str, default=None, help='恢复文件路径（progress.json或final_result.json）')
    parser.add_argument('--resume-best-only', action='store_true', help='只恢复最佳个体（需配合--resume-file）')

    return parser.parse_args()


def main():
    """主函数"""
    # 设置日志级别：显示 INFO，但过滤掉因子计算失败的 WARNING
    from core.logger import core_logger

    def filter_factor_warnings(record):
        """过滤掉因子计算失败的 WARNING"""
        # 如果是 WARNING 级别，且消息包含"因子"和"计算失败"，则过滤掉
        if record["level"].name == "WARNING":
            message = record["message"]
            if "因子" in message and ("计算失败" in message or "计算错误" in message):
                return False  # 过滤掉
        return True  # 保留其他所有日志

    core_logger.real_logger.remove()
    core_logger.real_logger.add(
        sink=sys.stdout,
        format=core_logger.log_format,
        level="INFO",  # 显示 INFO 及以上
        filter=filter_factor_warnings  # 但过滤掉因子计算的 WARNING
    )

    args = parse_arguments()

    # 参数验证
    if args.resume_best_only and not args.resume_file:
        testback_logger.error("❌ 参数错误: --resume-best-only 需要配合 --resume-file 使用")
        sys.exit(1)

    # 创建结果目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if args.resume_file:
        # Resume模式：继承输出目录
        resume_path = Path(args.resume_file)
        if resume_path.name == 'progress.json':
            result_dir = str(resume_path.parent)
        else:
            result_dir = str(resume_path.parent)
        testback_logger.info(f"📂 继承输出目录: {result_dir}")
    else:
        # 创建新的输出目录
        result_dir = f"results/ga_{timestamp}"
        os.makedirs(result_dir, exist_ok=True)
        testback_logger.info(f"📂 创建新输出目录: {result_dir}")

    # 输出配置
    max_workers = args.max_workers if args.max_workers else args.population
    testback_logger.info("="*60)
    testback_logger.info("GA优化 - 多目标遗传算法")
    testback_logger.info("="*60)
    testback_logger.info(f"优化目标: 年化超额收益↑ + 夏普比率↑ + Calmar比率↑")
    testback_logger.info(f"种群: {args.population}, 代数: {args.generations}")
    testback_logger.info(f"回测: {args.backtest_days}天, 资金: {args.initial_capital:.0f}")
    testback_logger.info(f"选股数量: {args.rank_n}")
    testback_logger.info(f"并行进程数: {max_workers}")
    testback_logger.info(f"结果目录: {result_dir}")
    testback_logger.info("="*60)

    # 因子范围（从当前配置读取）
    factor_ranges = {factor: (-1.0, 1.0) for factor in FactorWeights.keys()}
    testback_logger.info(f"优化因子: {list(factor_ranges.keys())}")

    # 恢复模式
    previous_best_genes = None
    initial_population_genes = None
    start_generation = 0
    inherited_progress = None

    if args.resume_file:
        progress_info = load_progress_info(args.resume_file)
        start_generation = progress_info['current_generation'] + 1
        inherited_progress = progress_info['all_progress']
        testback_logger.info(f"✅ 继承种群: 从第{start_generation}代继续")

        if args.resume_best_only:
            previous_best_genes = load_previous_best(args.resume_file, factor_ranges)
            testback_logger.info("✅ 恢复模式: 只恢复最佳个体")
        else:
            initial_population_genes = load_population_genes(
                args.resume_file,
                args.population,
                factor_ranges
            )
            testback_logger.info(f"✅ 恢复模式: 整个种群({len(initial_population_genes)}个个体)")
    else:
        testback_logger.info("🎲 初始化模式: 所有因子权重初始化为1.0")
        previous_best_genes = {factor_name: 1.0 for factor_name in factor_ranges.keys()}

    # 准备数据（训练集/验证集/测试集）
    cache_keys = prepare_data(
        backtest_days=args.backtest_days,
        stock_limit=args.stock_limit,
        use_random_sample=True  # 使用随机采样训练集
    )

    # 适应度函数参数（使用训练集）
    fitness_params = {
        'topn_cache_key': cache_keys['train'],  # GA 在训练集上优化
        'initial_cash': args.initial_capital,
        'rank_n': args.rank_n,
        'commission_rate': 0.0003,
        'min_commission': 5.0
    }

    # 创建GA实例
    testback_logger.info("\n" + "="*60)
    testback_logger.info("创建GA优化器...")
    testback_logger.info("="*60)

    ga = NSGA2GeneticAlgorithm(
        factor_ranges=factor_ranges,
        fitness_function=evaluate_individual,
        fitness_params=fitness_params,
        population_size=args.population,
        mutation_rate=0.2,
        crossover_rate=0.8,
        max_generations=args.generations,
        max_workers=max_workers,
        progress_callback=lambda gen, pf, g: save_progress(
            gen, g, result_dir, inherited_progress=inherited_progress
        ),
        initial_genes=previous_best_genes,
        initial_population_genes=initial_population_genes,
        start_generation=start_generation,
        result_dir=result_dir,
        elite_pool_size=args.population
    )

    # 运行GA
    testback_logger.info("\n" + "="*60)
    testback_logger.info("开始GA优化...")
    testback_logger.info("="*60)

    try:
        population, pareto_front = ga.run()
    except KeyboardInterrupt:
        testback_logger.warning("\n⚠️ 用户中断优化")
        return

    # 输出最终结果
    testback_logger.info("\n" + "="*60)
    testback_logger.info("GA优化完成（训练集）")
    testback_logger.info("="*60)

    # === 在验证集和测试集上评估 Pareto 前沿 ===
    testback_logger.info("\n" + "="*60)
    testback_logger.info("在验证集和测试集上评估 Pareto 前沿...")
    testback_logger.info("="*60)

    def evaluate_on_dataset(genes, dataset_name, cache_key):
        """在指定数据集上评估权重"""
        testback_logger.info(f"\n评估数据集: {dataset_name}")
        result = evaluate_individual(
            genes=genes,
            topn_cache_key=cache_key,
            initial_cash=args.initial_capital,
            rank_n=args.rank_n,
            commission_rate=0.0003,
            min_commission=5.0,
            individual_idx=0,
            generation=999
        )
        excess, sharpe, calmar, metrics = result
        testback_logger.info(
            f"  {dataset_name} 结果: 超额={excess:+.4f} 夏普={sharpe:.2f} Calmar={calmar:.2f} | "
            f"年化={metrics['annualized_return']:.4f} 回撤={metrics['max_drawdown']:.4f}"
        )
        return {
            'excess': excess,
            'sharpe': sharpe,
            'calmar': calmar,
            'metrics': metrics
        }

    # 评估 Pareto 前沿中的最佳个体（前3个）
    pareto_evaluation = []
    for idx, ind in enumerate(pareto_front[:3]):
        testback_logger.info(f"\n--- Pareto 个体 [{idx}] ---")
        testback_logger.info(f"权重: {ind.genes}")

        # 训练集（已有）
        train_result = {
            'excess': ind.fitness.values[0],
            'sharpe': ind.fitness.values[1],
            'calmar': ind.fitness.values[2],
            'metrics': ind.detailed_metrics
        }
        testback_logger.info(
            f"  训练集: 超额={train_result['excess']:+.4f} 夏普={train_result['sharpe']:.2f} Calmar={train_result['calmar']:.2f}"
        )

        # 验证集
        val_result = evaluate_on_dataset(ind.genes, "验证集", cache_keys['val'])

        # 测试集
        test_result = evaluate_on_dataset(ind.genes, "测试集", cache_keys['test'])

        pareto_evaluation.append({
            'genes': ind.genes,
            'train': train_result,
            'val': val_result,
            'test': test_result
        })

    testback_logger.info("\n" + "="*60)

    # 保存最终结果
    pareto_individuals = []
    for ind in pareto_front:
        pareto_individuals.append({
            'genes': ind.genes,
            'objective1_annualized_excess': ind.fitness.values[0],
            'objective2_sharpe_ratio': ind.fitness.values[1],
            'objective3_calmar_ratio': ind.fitness.values[2],
            **ind.detailed_metrics
        })

    final_result = {
        'timestamp': timestamp,
        'mode': 'multi_objective',
        'optimization_targets': ['annualized_excess', 'sharpe_ratio', 'calmar_ratio'],
        'config': {
            'population': args.population,
            'generations': args.generations,
            'backtest_days': args.backtest_days,
            'initial_capital': args.initial_capital,
            'rank_n': args.rank_n,
            'max_workers': max_workers,
            'stock_limit': args.stock_limit,
            'use_random_sample': True,
            'train_dates': '2022-2023 random sample',
            'val_dates': '2024-01-01 to 2025-01-01',
            'test_dates': '2025-01-01 to today'
        },
        'pareto_front': pareto_individuals,
        'pareto_front_size': len(pareto_front),
        'pareto_evaluation_top3': pareto_evaluation,  # 新增：验证集和测试集结果
        'total_time_seconds': sum(ga.generation_time_history) if ga.generation_time_history else 0
    }

    # 保存final_result.json
    final_result_file = os.path.join(result_dir, 'final_result.json')
    with open(final_result_file, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)
    testback_logger.info(f"✅ 最终结果已保存: {final_result_file}")

    # 输出Pareto前沿
    testback_logger.info(f"\n📊 Pareto前沿: {len(pareto_front)}个个体")
    testback_logger.info("="*60)
    for idx, ind in enumerate(pareto_front[:10]):
        obj1, obj2, obj3 = ind.fitness.values
        testback_logger.info(
            f"  [{idx}] 超额={obj1:+.4f} 夏普={obj2:.2f} Calmar={obj3:.2f}"
        )
    testback_logger.info("="*60)

    # 时间统计
    if ga.generation_time_history:
        total_time = sum(ga.generation_time_history)
        avg_time = total_time / len(ga.generation_time_history)
        testback_logger.info("\n时间统计:")
        testback_logger.info(f"  总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
        testback_logger.info(f"  平均每代: {avg_time:.1f}秒")
        testback_logger.info(f"  最快一代: {min(ga.generation_time_history):.1f}秒")
        testback_logger.info(f"  最慢一代: {max(ga.generation_time_history):.1f}秒")

    testback_logger.info("\n✅ 程序正常结束")


if __name__ == "__main__":
    main()
