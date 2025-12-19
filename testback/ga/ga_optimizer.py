"""
NSGA-II多目标遗传算法优化器

特性：
1. 多目标优化（年化超额收益、夏普比率、Calmar比率等）
2. 历史最优个体选择机制
3. 每代评估3k个体（k父代 + k子代 + k历史最优）
4. 多进程并行评估
5. 进度保存与恢复

参考：qmt-trade/ga.py
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Any, Callable
import time
from joblib import Parallel, delayed, parallel_backend
from deap import base, creator, tools
from pathlib import Path
import json
import os

from testback.logger import testback_logger


class NSGA2GeneticAlgorithm:
    """NSGA-II多目标遗传算法类

    优化目标：
    1. 年化超额收益率 ↑
    2. 夏普比率 ↑
    3. Calmar比率 ↑
    """

    def __init__(self,
                 factor_ranges: Dict[str, tuple],
                 fitness_function: Callable,
                 fitness_params: Dict = None,
                 population_size: int = 30,
                 mutation_rate: float = 0.2,
                 crossover_rate: float = 0.8,
                 max_generations: int = 50,
                 factor_configs: Dict = None,
                 max_workers: int = None,
                 progress_callback: Callable = None,
                 initial_genes: Dict = None,
                 initial_population_genes: List[Dict] = None,
                 start_generation: int = 0,
                 result_dir: str = None,
                 elite_pool_size: int = None):
        """初始化遗传算法

        Args:
            factor_ranges: 因子权重范围 {'MACD': (0, 2), 'BBI': (0, 2), ...}
            fitness_function: 适应度函数
            fitness_params: 适应度函数参数
            population_size: 种群大小（k）
            mutation_rate: 变异率
            crossover_rate: 交叉率
            max_generations: 最大代数
            factor_configs: 因子配置（离散值等）
            max_workers: 最大进程数
            progress_callback: 进度回调函数
            initial_genes: 初始基因（用于热启动）
            initial_population_genes: 初始种群基因列表（用于恢复）
            start_generation: 起始代数（用于恢复）
            result_dir: 结果输出目录
            elite_pool_size: 精英池大小（默认等于population_size）
        """
        self.factor_ranges = factor_ranges
        self.fitness_function = fitness_function
        self.fitness_params = fitness_params or {}
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.factor_configs = factor_configs or {}
        self.max_workers = max_workers
        self.progress_callback = progress_callback
        self.initial_genes = initial_genes
        self.initial_population_genes = initial_population_genes
        self.start_generation = start_generation
        self.result_dir = result_dir

        # 精英池大小（默认等于种群大小）
        self.elite_pool_size = elite_pool_size if elite_pool_size is not None else population_size

        self.pareto_fronts = []
        self.generation_time_history = []
        self.population = []
        self.all_evaluated_individuals = []  # 存储每代所有评估的个体

        self._setup_deap()

    def _setup_deap(self):
        """设置DEAP框架"""
        # 清理旧的定义
        if hasattr(creator, "FitnessMulti"):
            del creator.FitnessMulti
        if hasattr(creator, "Individual"):
            del creator.Individual

        # 多目标优化：3个目标都是最大化
        # (年化超额收益率, 夏普比率, Calmar比率)
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti, genes=dict)

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", tools.selNSGA2)

    def _create_individual(self):
        """创建一个个体"""
        ind = creator.Individual()
        genes = {}

        for factor_name, (min_val, max_val) in self.factor_ranges.items():
            # 检查是否有离散值配置
            if factor_name in self.factor_configs:
                discrete_values = self.factor_configs[factor_name].get('discrete_values')
                if discrete_values:
                    genes[factor_name] = random.choice(discrete_values)
                    continue

            # 连续值
            genes[factor_name] = random.uniform(min_val, max_val)

        ind.genes = genes
        ind.extend([0, 0, 0])  # 预留3个目标值的位置
        ind.detailed_metrics = {}

        return ind

    def _crossover(self, ind1, ind2):
        """交叉操作"""
        if random.random() > self.crossover_rate:
            return ind1, ind2

        child1_genes = {}
        child2_genes = {}

        for factor_name in ind1.genes.keys():
            if random.random() < 0.5:
                child1_genes[factor_name] = ind1.genes[factor_name]
                child2_genes[factor_name] = ind2.genes[factor_name]
            else:
                child1_genes[factor_name] = ind2.genes[factor_name]
                child2_genes[factor_name] = ind1.genes[factor_name]

        ind1.genes = child1_genes
        ind2.genes = child2_genes

        return ind1, ind2

    def _mutate(self, individual):
        """变异操作"""
        for factor_name in individual.genes.keys():
            if random.random() < self.mutation_rate:
                # 检查离散值
                if factor_name in self.factor_configs:
                    discrete_values = self.factor_configs[factor_name].get('discrete_values')
                    if discrete_values:
                        individual.genes[factor_name] = random.choice(discrete_values)
                        continue

                # 连续值
                min_val, max_val = self.factor_ranges[factor_name]
                individual.genes[factor_name] = random.uniform(min_val, max_val)

        return (individual,)

    def _evaluate_population(self, population: List, generation: int) -> List:
        """并行评估种群

        Args:
            population: 个体列表
            generation: 代数

        Returns:
            评估后的个体列表
        """
        t_gen_start = time.time()

        testback_logger.info(f"[第{generation}代] 开始并行评估{len(population)}个个体...")

        # 并行评估（使用 joblib + loky backend）
        t_eval_start = time.time()
        with parallel_backend('loky', n_jobs=self.max_workers, inner_max_num_threads=1):
            results = Parallel(
                n_jobs=self.max_workers,
                prefer='processes',
                batch_size=1,
            )(
                delayed(_evaluate_individual_task)(
                    ind, self.fitness_function, self.fitness_params, idx, generation
                )
                for idx, ind in enumerate(population)
            )
        t_eval_end = time.time()

        testback_logger.info(f"[第{generation}代] 并行评估耗时: {(t_eval_end-t_eval_start):.1f}秒")

        # 更新适应度
        for ind, result in zip(population, results):
            # result = (annualized_excess, sharpe_ratio, calmar_ratio, detailed_metrics)
            ind.fitness.values = result[:3]  # 3个目标
            ind.detailed_metrics = result[3]

        # 打印每个个体的权重（简化格式）
        testback_logger.info(f"[第{generation}代] 个体权重:")
        for idx, ind in enumerate(population):
            weights = {k: round(v, 2) for k, v in ind.genes.items()}
            obj1, obj2, obj3 = ind.fitness.values
            testback_logger.info(
                f"  [{idx:2d}] 超额={obj1:+.4f} 夏普={obj2:.2f} Calmar={obj3:.2f} | "
                f"{json.dumps(weights, ensure_ascii=False)}"
            )

        t_gen_total = time.time() - t_gen_start
        testback_logger.info(f"[第{generation}代] 评估完成 - 总耗时: {t_gen_total:.1f}秒")

        return population

    def _extract_all_elite_from_progress(self, progress_file: str) -> List[Dict]:
        """从progress.json提取所有历史个体（按平均超额收益排序）

        Args:
            progress_file: progress.json文件路径

        Returns:
            所有去重后的个体列表，按平均超额收益降序排序
        """
        if not os.path.exists(progress_file):
            return []

        with open(progress_file, 'r', encoding='utf-8') as f:
            progress_data = json.load(f)

        # 提取所有个体（过滤异常数据）
        all_individuals = []
        filtered_count = 0
        for gen_data in progress_data.get('generations', []):
            for ind in gen_data.get('all_individuals', []):
                # 过滤掉无效数据
                obj1 = ind.get('objective1_annualized_excess', 0)
                if not np.isfinite(obj1):
                    filtered_count += 1
                    continue

                all_individuals.append({
                    'genes': ind['genes'],
                    'objective1': obj1
                })

        if filtered_count > 0:
            testback_logger.info(f"Elite pool选择: 过滤掉 {filtered_count} 个无效个体")

        if not all_individuals:
            return []

        # 按基因去重，计算平均超额收益
        from collections import defaultdict
        gene_groups = defaultdict(list)

        for ind in all_individuals:
            gene_key = tuple(sorted(ind['genes'].items()))
            gene_groups[gene_key].append(ind['objective1'])

        # 计算每个基因的平均超额收益
        gene_avg_scores = []
        for gene_key, scores in gene_groups.items():
            avg_score = np.mean(scores)
            genes = dict(gene_key)
            gene_avg_scores.append({
                'genes': genes,
                'avg_excess': avg_score,
                'count': len(scores),
                'min': min(scores),
                'max': max(scores)
            })

        # 排序
        gene_avg_scores.sort(key=lambda x: x['avg_excess'], reverse=True)

        testback_logger.info(
            f"📊 历史个体统计: 总个体数={len(all_individuals)}, "
            f"去重后={len(gene_avg_scores)}"
        )

        return gene_avg_scores

    def _create_individual_from_genes(self, genes: Dict):
        """从基因字典创建个体"""
        ind = creator.Individual()
        ind.genes = genes.copy()
        ind.extend([0, 0, 0])
        ind.detailed_metrics = {}
        return ind

    def _get_elite_from_history(self, exclude_population: List, count: int) -> List:
        """从历史记录中提取k个最优个体（按平均超额收益排序，剔除重复）

        Args:
            exclude_population: 需要排除的个体列表（父代+子代）
            count: 需要提取的个体数量

        Returns:
            精英个体列表
        """
        progress_file = os.path.join(self.result_dir, 'progress.json')
        all_elite_data = self._extract_all_elite_from_progress(progress_file)

        if not all_elite_data:
            return []

        # 构建排除集合
        exclude_genes_set = set()
        for ind in exclude_population:
            gene_key = tuple(sorted(ind.genes.items()))
            exclude_genes_set.add(gene_key)

        # 选取不重复的前count个
        elite_individuals = []
        for elite in all_elite_data:
            gene_key = tuple(sorted(elite['genes'].items()))
            if gene_key not in exclude_genes_set:
                ind = self._create_individual_from_genes(elite['genes'])
                elite_individuals.append(ind)
                if len(elite_individuals) >= count:
                    break

        return elite_individuals

    def _generate_offspring(self, parents: List) -> List:
        """从父代生成子代（选择+交叉+变异）

        Args:
            parents: 父代个体列表（k个）

        Returns:
            子代个体列表（k个）
        """
        k = len(parents)

        # 生成子代
        offspring = []
        for _ in range(k):
            # 随机选择两个父代进行交叉
            p1, p2 = random.sample(parents, 2)
            child = self.toolbox.clone(p1)

            # 交叉
            if random.random() < self.crossover_rate:
                _, child = self._crossover(self.toolbox.clone(p1), self.toolbox.clone(p2))

            # 变异
            self._mutate(child)

            offspring.append(child)

        return offspring

    def run(self):
        """运行遗传算法

        流程：
        - 第0代：生成3k个随机个体 → 评估3k → 选择k个父代
        - 第N代：准备3k个体（k父代+k子代+k历史最优）→ 评估3k → 选择k个父代
        - Resume模式：从start_generation继续，跳过第0代初始化
        """
        k = self.population_size
        total_generations = self.start_generation + self.max_generations
        testback_logger.info(
            f"开始GA优化: k={k}, 每代评估={3*k}, "
            f"起始代数={self.start_generation}, 运行代数={self.max_generations}, "
            f"结束代数={total_generations-1}"
        )

        # Resume模式：跳过第0代，直接使用继承的种群
        if self.start_generation > 0 and self.initial_population_genes:
            testback_logger.info(f"✅ Resume模式：跳过初始化，从第{self.start_generation}代继续")
            parents = []
            for genes in self.initial_population_genes[:k]:
                ind = self.toolbox.individual()
                ind.genes = genes.copy()
                parents.append(ind)

            # 如果继承的种群不足k个，用随机个体补充
            if len(parents) < k:
                shortfall = k - len(parents)
                testback_logger.info(f"继承种群不足，随机补充 {shortfall} 个")
                parents.extend(self.toolbox.population(n=shortfall))

            self.population = parents
        else:
            # ==================== 第0代 ====================
            first_gen = self.start_generation
            gen_start_time = time.time()
            testback_logger.info(f"\n{'='*60}")
            testback_logger.info(f"代数 {first_gen}/{total_generations-1}")

            # 生成3k个随机个体
            all_individuals = self.toolbox.population(n=3*k)
            testback_logger.info(f"生成 {len(all_individuals)} 个随机个体")

            # 评估3k个
            all_individuals = self._evaluate_population(all_individuals, generation=first_gen)

            # 保存所有评估的个体
            self.all_evaluated_individuals = all_individuals

            # 选择k个作为父代
            parents = self.toolbox.select(all_individuals, k)
            self.population = parents

            # 记录
            pareto_front = tools.sortNondominated(parents, len(parents), first_front_only=True)[0]
            self.pareto_fronts.append(pareto_front)
            gen_time = time.time() - gen_start_time
            self.generation_time_history.append(gen_time)

            self._log_generation_stats(first_gen, parents, pareto_front, gen_time)

            if self.progress_callback:
                self.progress_callback(first_gen, pareto_front, self)

        # ==================== 后续代 ====================
        loop_start = self.start_generation + 1 if self.start_generation == 0 or not self.initial_population_genes else self.start_generation
        for gen in range(loop_start, total_generations):
            gen_start_time = time.time()
            testback_logger.info(f"\n{'='*60}")
            testback_logger.info(f"代数 {gen}/{total_generations-1}")

            # 准备3k个体
            # 1. k个父代
            # 2. k个子代
            offspring = self._generate_offspring(parents)

            # 3. k个历史最优
            elite = self._get_elite_from_history(
                exclude_population=parents + offspring,
                count=k
            )

            # 如果历史最优不足k个，用随机个体补充
            if len(elite) < k:
                shortfall = k - len(elite)
                testback_logger.info(f"历史最优不足，随机补充 {shortfall} 个")
                elite.extend(self.toolbox.population(n=shortfall))

            # 合并3k个体
            all_individuals = list(parents) + offspring + elite
            testback_logger.info(
                f"准备评估: {len(parents)}父代 + {len(offspring)}子代 + "
                f"{len(elite)}历史最优 = {len(all_individuals)}个"
            )

            # 清除所有个体的fitness（因为每代回测条件可能不同）
            for ind in all_individuals:
                if hasattr(ind.fitness, 'values') and ind.fitness.valid:
                    del ind.fitness.values

            # 一次性并行评估3k个
            all_individuals = self._evaluate_population(all_individuals, generation=gen)

            # 保存所有评估的个体
            self.all_evaluated_individuals = all_individuals

            # 选择k个作为下一代父代
            parents = self.toolbox.select(all_individuals, k)
            self.population = parents

            # 记录
            pareto_front = tools.sortNondominated(parents, len(parents), first_front_only=True)[0]
            self.pareto_fronts.append(pareto_front)
            gen_time = time.time() - gen_start_time
            self.generation_time_history.append(gen_time)

            self._log_generation_stats(gen, parents, pareto_front, gen_time)

            if self.progress_callback:
                self.progress_callback(gen, pareto_front, self)

        # 完成
        testback_logger.info(f"\nGA优化完成")

        return self.population, self.pareto_fronts[-1]

    def _log_generation_stats(self, gen: int, population: List, pareto_front: List, gen_time: float):
        """输出代统计信息"""
        testback_logger.info(f"代数 {gen} | 耗时: {gen_time:.1f}s")
        testback_logger.info(f"📊 Pareto前沿: {len(pareto_front)}个个体")

        for idx, ind in enumerate(pareto_front[:5]):
            obj1, obj2, obj3 = ind.fitness.values
            testback_logger.info(
                f"  [{idx}] 超额={obj1:+.4f} 夏普={obj2:.2f} Calmar={obj3:.2f}"
            )

        avg_obj1 = np.mean([ind.fitness.values[0] for ind in population])
        avg_obj2 = np.mean([ind.fitness.values[1] for ind in population])
        avg_obj3 = np.mean([ind.fitness.values[2] for ind in population])
        testback_logger.info(
            f"  📊 种群平均: 超额={avg_obj1:+.4f} 夏普={avg_obj2:.2f} Calmar={avg_obj3:.2f}"
        )

    def get_pareto_front(self):
        """获取最终的Pareto前沿"""
        if self.pareto_fronts:
            return self.pareto_fronts[-1]
        return []


# ============================================================================
# 模块级函数：用于并行评估（joblib worker）
# ============================================================================

def _evaluate_individual_task(ind, fitness_func, fitness_params, idx, gen):
    """并行评估单个个体（worker函数）"""
    fitness_params = fitness_params.copy()
    fitness_params['individual_idx'] = idx
    fitness_params['generation'] = gen

    return fitness_func(ind.genes, **fitness_params)
