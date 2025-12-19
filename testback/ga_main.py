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

from testback.logger import testback_logger
from testback.ga import NSGA2GeneticAlgorithm, load_progress_info, load_population_genes, save_progress
from testback.account import StockAccountMocker
from core import init_stock_detail_cache
from core.database import allow_buy_stock_code_list, init_full_data
from core.strategies import TopN
from core.strategies._weights import FactorWeights, RANK_N
from utils.stock.time import get_trading_date_span
from utils.parallel import batch_run_threads
from utils.shared_memory import SharedMemoryCache

# 全局缓存
ga_cache = SharedMemoryCache('testback_cache', compress_level=6)

def _evaluate_individual_worker(
    genes: Dict[str, float],
    topn_cache_key: str,
    initial_cash: float = 500_000.0,
    rank_n: int = RANK_N,
    commission_rate: float = 0.0003,
    min_commission: float = 5.0,
    individual_idx: int = 0,
    generation: int = 0
) -> Tuple[float, float, float, Dict]:
    """独立进程评估个体 - 从共享内存读取数据（完全对齐 main.py）"""
    t_start = time.time()

    try:
        # 延迟导入：每个 worker 只导入自己需要的模块
        from xtquant import xtdata
        xtdata.enable_hello = False
        from core.strategies.sizers.sizer import Sizer
        from core import get_market_data_from_cache
        from utils.stock.info import get_baseline_data

        # 从共享内存读取数据
        topn_list = ga_cache.get(topn_cache_key)
        if not topn_list:
            testback_logger.error(f"[个体{individual_idx}] 无法从共享内存读取数据")
            return (-999.0, -999.0, -999.0, {})

        # 创建账户模拟器
        account = StockAccountMocker(cash=initial_cash, commission=commission_rate, min_commission=min_commission)

        # 记录上一日持仓
        prev_holdings = set()

        # 回测统计
        daily_returns = []
        baseline_returns = []

        # 遍历每个交易日
        for topn in topn_list:
            trade_date = topn.base_date

            # 1. 获取当日 Top N 股票（使用传入的权重）
            top_stocks = topn.get_ordered_stocks(
                n=rank_n,
                weights=genes,
                temperatures={k: 1.0 for k in genes.keys()},
                norm_method='rank'
            )

            if not top_stocks:
                daily_returns.append(0.0)
                baseline_returns.append(0.0)
                continue

            # 2. 获取股票价格
            prices = {}
            for stock in top_stocks:
                try:
                    trade_datetime = datetime.combine(trade_date, datetime.min.time())
                    data = get_market_data_from_cache(stock, 1, trade_datetime)
                    if data is not None and len(data) > 0:
                        prices[stock] = float(data.iloc[-1]['close'])
                except Exception as e:
                    testback_logger.debug(f"{stock} 价格获取失败: {e}")

            if not prices:
                daily_returns.append(0.0)
                baseline_returns.append(0.0)
                continue

            # 3. 计算仓位分配
            target_holdings = set(top_stocks)
            allocations = Sizer.allocate(stocks=top_stocks, total_capital=account.current_cash, prices=prices)

            # 4. 卖出不在目标持仓中的股票
            stocks_to_sell = prev_holdings - target_holdings
            for stock in stocks_to_sell:
                pos = account.get_position(stock)
                if pos and pos['volume'] > 0:
                    try:
                        sell_price = prices.get(stock) or pos['avg_price']
                        account.sell_stock(code=stock, volume=pos['volume'], price=sell_price, sell_date=trade_date, clear_reason='调仓')
                    except Exception as e:
                        testback_logger.debug(f"卖出 {stock} 失败: {e}")

            # 5. 买入新股票
            for stock, shares in allocations.items():
                if shares <= 0 or account.get_position(stock):
                    continue  # 跳过无效分配或已持仓

                try:
                    account.buy_stock(code=stock, volume=shares, price=prices[stock], buy_date=trade_date)
                except Exception as e:
                    testback_logger.debug(f"买入 {stock} 失败: {e}")

            # 6. 记录当日资产并计算收益率
            trade_datetime = datetime.combine(trade_date, datetime.min.time())
            assets = account.calc_assets(trade_datetime)

            # 计算日收益率
            prev_date_list = sorted(account.daily_assets.keys())
            if len(prev_date_list) >= 2:
                prev_value = account.daily_assets[prev_date_list[-2]]
                daily_return = (assets['total_asset'] - prev_value) / prev_value if prev_value > 0 else 0.0
            else:
                daily_return = 0.0

            daily_returns.append(daily_return)

            # 计算基准收益
            baseline_data = get_baseline_data(trade_datetime)
            baseline_return = 0.0
            if baseline_data is not None and len(prev_date_list) >= 2:
                prev_baseline = get_baseline_data(prev_date_list[-2])
                if prev_baseline is not None:
                    baseline_return = (float(baseline_data['close']) - float(prev_baseline['close'])) / float(prev_baseline['close'])

            baseline_returns.append(baseline_return)

            # 更新持仓记录
            prev_holdings = target_holdings

        # 7. 计算最终收益
        final_assets = account.calc_assets(datetime.combine(topn_list[-1].base_date, datetime.min.time()))
        total_return = (final_assets['total_asset'] - initial_cash) / initial_cash if initial_cash > 0 else 0.0

        # 计算回测指标
        trading_days = len(daily_returns)
        annualized_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0.0
        baseline_annualized = (1 + sum(baseline_returns)) ** (252 / trading_days) - 1 if trading_days > 0 else 0.0
        annualized_excess = annualized_return - baseline_annualized

        # 最大回撤
        cumulative_values = [initial_cash]
        for ret in daily_returns:
            cumulative_values.append(cumulative_values[-1] * (1 + ret))

        max_drawdown = 0.0
        peak = cumulative_values[0]
        for value in cumulative_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0.0
            max_drawdown = max(max_drawdown, drawdown)

        # 夏普比率
        if trading_days > 1:
            returns_std = np.std(daily_returns) * np.sqrt(252)
            sharpe_ratio = (annualized_return - 0.03) / returns_std if returns_std > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        # Calmar比率
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0

        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'baseline_return': baseline_annualized,
            'annualized_excess': annualized_excess,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'total_trades': len(account.cleared_positions),
            'trading_days': trading_days
        }

        # 输出日志
        testback_logger.info(
            f"[个体{individual_idx}] 超额={annualized_excess:+.4f} 夏普={sharpe_ratio:.2f} Calmar={calmar_ratio:.2f} | "
            f"年化={annualized_return:.4f} 基准={baseline_annualized:.4f} 回撤={max_drawdown:.4f} | "
            f"交易={metrics['total_trades']}次 耗时={time.time()-t_start:.1f}秒"
        )

        return (annualized_excess, sharpe_ratio, calmar_ratio, metrics)

    except Exception as e:
        testback_logger.error(f"[个体{individual_idx}] 回测时出错: {e}")
        import traceback
        testback_logger.error(traceback.format_exc())
        return (-999.0, -999.0, -999.0, {})


if __name__ == "__main__":
    # 设置日志级别（过滤因子计算警告）
    from core.logger import core_logger

    def filter_factor_warnings(record):
        if record["level"].name == "WARNING" and "因子" in record["message"] and "计算失败" in record["message"]:
            return False
        return True

    core_logger.real_logger.remove()
    core_logger.real_logger.add(sink=sys.stdout, format=core_logger.log_format, level="INFO", filter=filter_factor_warnings)

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
    worker_count = min(os.cpu_count() or 4, args.population)

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

        topNs = batch_run_threads(
            func=TopN,
            args_list=[[all_stocks, datetime.combine(d, datetime.min.time())] for d in dates],
            max_workers=8,
        )
        cache_key = f"topn_{name}_{timestamp}"
        ga_cache.put(cache_key, topNs)
        cache_keys[name] = cache_key

    # 定义每代回调：保存进度 + 评估验证/测试集
    def _generation_callback(gen, pareto_front, ga_instance):
        """每代结束后：保存进度 + 评估Pareto前沿在验证/测试集的表现"""
        # 1. 保存进度
        save_progress(gen, ga_instance, result_dir, inherited_progress=inherited_progress)

        # 2. 评估Pareto前沿的top个体（最多3个）
        top_n = min(3, len(pareto_front))
        if top_n == 0:
            return

        testback_logger.info(f"\n{'='*60}")
        testback_logger.info(f"[第{gen}代] Pareto前沿Top{top_n}在三个数据集的表现:")

        for idx in range(top_n):
            ind = pareto_front[idx]
            train_excess, train_sharpe, train_calmar = ind.fitness.values

            testback_logger.info(f"\n  [{idx}] 训练集: 超额={train_excess:+.4f} 夏普={train_sharpe:.2f} Calmar={train_calmar:.2f}")

            # 验证集评估
            val_excess, val_sharpe, val_calmar, _ = _evaluate_individual_worker(
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
            test_excess, test_sharpe, test_calmar, _ = _evaluate_individual_worker(
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

        testback_logger.info(f"{'='*60}")

    # 创建并运行GA
    testback_logger.info("开始GA优化...")
    ga = NSGA2GeneticAlgorithm(
        factor_ranges=factor_ranges,
        fitness_function=_evaluate_individual_worker,
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
                excess, sharpe, calmar, metrics = _evaluate_individual_worker(
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
