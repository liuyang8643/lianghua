"""
GA工具函数

包含：
1. 进度保存/加载
2. 最佳个体加载
3. 种群基因加载
4. 结果分析
"""

import os
import json
import numpy as np
from typing import Dict, List
from datetime import datetime
import psutil

from testback.logger import testback_logger


def load_previous_best(resume_file: str = None, factor_ranges: Dict = None) -> Dict:
    """加载上次运行的最佳参数

    Args:
        resume_file: 恢复文件路径
        factor_ranges: 因子范围（用于过滤）

    Returns:
        最佳基因字典
    """
    if not resume_file or not os.path.exists(resume_file):
        testback_logger.warning(f"恢复文件不存在: {resume_file}")
        return {}

    testback_logger.info(f"正在加载: {resume_file}")

    with open(resume_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 尝试从不同的数据结构中提取基因
    if 'best_individual' in data and 'genes' in data['best_individual']:
        genes = data['best_individual']['genes']
        fitness = data['best_individual'].get('fitness', 'N/A')
        testback_logger.info(f"加载的最佳参数适应度: {fitness}")
    else:
        # 直接从根级别提取
        metadata_fields = {'fitness', 'strategy_return', 'benchmark_return', 'generation', 'timestamp'}
        genes = {k: v for k, v in data.items() if k not in metadata_fields}
        fitness = data.get('fitness', 'N/A')
        testback_logger.info(f"加载的最佳参数适应度: {fitness}")
        testback_logger.info(f"加载了{len(genes)}个基因参数")

    # 过滤因子
    if factor_ranges:
        genes = {k: v for k, v in genes.items() if k in factor_ranges}

    return genes


def load_progress_info(resume_file: str) -> Dict:
    """从progress.json加载进度信息

    Args:
        resume_file: progress.json文件路径

    Returns:
        进度信息字典
    """
    with open(resume_file, 'r', encoding='utf-8') as f:
        progress_data = json.load(f)

    current_gen = progress_data['current_generation']

    testback_logger.info(f"从进度文件加载: {resume_file}")
    testback_logger.info(f"已完成代数: {current_gen}")

    return {
        'current_generation': current_gen,
        'all_progress': progress_data
    }


def load_population_genes(resume_file: str, population_size: int, factor_ranges: Dict = None) -> List[Dict]:
    """从指定json文件加载种群基因列表

    Args:
        resume_file: 恢复文件路径
        population_size: 种群大小
        factor_ranges: 因子范围（用于过滤）

    Returns:
        基因列表
    """
    testback_logger.info(f"正在加载种群基因: {resume_file}")

    with open(resume_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    genes_list = []

    # 从generations中加载
    if 'generations' in data and isinstance(data['generations'], list):
        generations = data['generations']
        testback_logger.info(f"发现{len(generations)}代历史数据")

        # 从最后一代向前遍历
        for gen_idx in range(len(generations) - 1, -1, -1):
            gen_data = generations[gen_idx]

            if 'all_individuals' not in gen_data:
                continue

            individuals = gen_data['all_individuals']
            gen_num = gen_data.get('generation', gen_idx)

            testback_logger.info(f"第{gen_num}代有{len(individuals)}个个体")

            for ind in individuals:
                genes = ind['genes'] if 'genes' in ind else ind

                # 过滤因子
                if factor_ranges:
                    genes = {k: v for k, v in genes.items() if k in factor_ranges}

                genes_list.append(genes)

                if len(genes_list) >= population_size:
                    testback_logger.info(f"✅ 成功加载{len(genes_list)}个个体基因(从第{gen_num}代及之前)")
                    return genes_list[:population_size]

        testback_logger.info(f"成功加载{len(genes_list)}个个体基因")
        return genes_list

    # 从best_individual加载
    elif 'best_individual' in data and 'genes' in data['best_individual']:
        genes = data['best_individual']['genes']

        if factor_ranges:
            genes = {k: v for k, v in genes.items() if k in factor_ranges}

        testback_logger.info("从final_result.json加载了1个最佳个体基因")
        return [genes]

    else:
        raise ValueError("文件格式不正确")


def save_progress(generation: int, ga, result_dir: str, inherited_progress: Dict = None):
    """实时保存进度

    Args:
        generation: 当前代数
        ga: GA实例
        result_dir: 结果目录
        inherited_progress: 继承的进度数据（用于resume）
    """
    progress_file = os.path.join(result_dir, 'progress.json')

    # 内存统计
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024

    # 所有评估的个体（3k个）
    all_evaluated = ga.all_evaluated_individuals
    population = ga.population  # 选中的父代（k个）

    from deap import tools
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

    # 保存所有评估的个体
    individuals_data = []
    for idx, ind in enumerate(all_evaluated):
        individuals_data.append({
            'individual_id': idx,
            'genes': ind.genes,
            'objective1_annualized_excess': ind.fitness.values[0],
            'objective2_sharpe_ratio': ind.fitness.values[1],
            'objective3_calmar_ratio': ind.fitness.values[2],
            **ind.detailed_metrics
        })

    # 保存Pareto前沿
    pareto_front_data = []
    for idx, ind in enumerate(pareto_front):
        # 查找该个体在种群中的索引
        population_idx = None
        for pop_idx, pop_ind in enumerate(population):
            if pop_ind is ind:
                population_idx = pop_idx
                break

        pareto_front_data.append({
            'individual_id': population_idx if population_idx is not None else idx,
            'genes': ind.genes,
            'objective1_annualized_excess': ind.fitness.values[0],
            'objective2_sharpe_ratio': ind.fitness.values[1],
            'objective3_calmar_ratio': ind.fitness.values[2],
            **ind.detailed_metrics
        })

    # 种群统计
    population_stats = {
        'mean_obj1': np.mean([ind.fitness.values[0] for ind in all_evaluated]),
        'std_obj1': np.std([ind.fitness.values[0] for ind in all_evaluated]),
        'max_obj1': np.max([ind.fitness.values[0] for ind in all_evaluated]),
        'min_obj1': np.min([ind.fitness.values[0] for ind in all_evaluated]),
        'mean_obj2': np.mean([ind.fitness.values[1] for ind in all_evaluated]),
        'std_obj2': np.std([ind.fitness.values[1] for ind in all_evaluated]),
        'mean_obj3': np.mean([ind.fitness.values[2] for ind in all_evaluated]),
        'std_obj3': np.std([ind.fitness.values[2] for ind in all_evaluated]),
        'pareto_front_size': len(pareto_front),
        'total_evaluated': len(all_evaluated),
        'selected_parents': len(population)
    }

    # 当前代数据
    current_gen_data = {
        'generation': generation,
        'population_size': len(all_evaluated),
        'selected_size': len(population),
        'generation_time': ga.generation_time_history[-1],
        'memory_mb': memory_mb,
        'timestamp': datetime.now().strftime("%Y%m%d %H:%M:%S"),
        'pareto_front': pareto_front_data,
        'population_stats': population_stats,
        'all_individuals': individuals_data
    }

    # 加载或创建进度数据
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            all_progress = json.load(f)
    elif inherited_progress:
        all_progress = inherited_progress.copy()
        testback_logger.info(f"继承{len(all_progress['generations'])}代历史数据")
    else:
        all_progress = {'generations': []}

    all_progress['generations'].append(current_gen_data)
    all_progress['current_generation'] = generation
    all_progress['total_time'] = sum(ga.generation_time_history) if ga.generation_time_history else 0
    all_progress['last_update'] = datetime.now().strftime("%Y%m%d %H:%M:%S")

    # 保存到文件
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(all_progress, f, indent=2, ensure_ascii=False)

    # 内存回收
    import gc
    gc.collect()
    testback_logger.info(f"内存使用: {memory_mb:.1f} MB")


def analyze_results(result_dir: str):
    """分析GA结果

    Args:
        result_dir: 结果目录
    """
    progress_file = os.path.join(result_dir, 'progress.json')
    if not os.path.exists(progress_file):
        testback_logger.error(f"进度文件不存在: {progress_file}")
        return

    with open(progress_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    testback_logger.info(f"\n{'='*60}")
    testback_logger.info("GA优化结果分析")
    testback_logger.info(f"{'='*60}")

    # 总体统计
    total_generations = data.get('current_generation', 0) + 1
    total_time = data.get('total_time', 0)
    testback_logger.info(f"总代数: {total_generations}")
    testback_logger.info(f"总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")

    # 最终Pareto前沿
    if 'final_summary' in data and 'pareto_front' in data['final_summary']:
        pareto_front = data['final_summary']['pareto_front']
        testback_logger.info(f"\nPareto前沿: {len(pareto_front)}个个体")

        for idx, ind in enumerate(pareto_front[:10]):
            testback_logger.info(
                f"  [{idx}] 超额={ind['objective1_annualized_excess']:.4f} "
                f"夏普={ind['objective2_sharpe_ratio']:.2f} "
                f"Calmar={ind['objective3_calmar_ratio']:.2f}"
            )

    # 收敛曲线
    testback_logger.info(f"\n收敛曲线:")
    generations = data.get('generations', [])
    for gen_data in generations:
        gen = gen_data['generation']
        stats = gen_data['population_stats']
        testback_logger.info(
            f"  第{gen:3d}代: 平均超额={stats['mean_obj1']:+.4f} "
            f"最大超额={stats['max_obj1']:+.4f} "
            f"Pareto前沿={stats['pareto_front_size']}个"
        )

    testback_logger.info(f"{'='*60}\n")
