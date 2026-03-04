import os
from pathlib import Path

from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm

from core import cleanup_shared_cache, get_market_data_from_cache, init_stock_detail_cache
from core.database import init_full_data
from utils.parallel import batch_run_threads
from utils.shared_memory import SharedMemoryCache
from testback.account import StockAccountMocker

os.environ['LOKY_PICKLER'] = 'pickle'  # 使用更快的pickle

from datetime import date, datetime, timedelta
from testback.logger import testback_logger
from core.strategies import TopN

# ========== GA 核心函数 ==========

# GA 搜索空间定义（离散值）
GA_SEARCH_SPACES = {
  'buy_n': [20, 30, 40],
  'sell_m': [20, 30, 40, 100, 200, 400, 600, 800],
  'temperatures': [1],
  'weights': [i/5 for i in range(-5, 6)]  # [-1.0, -0.8, -0.6, ..., 0.6, 0.8, 1.0]
}

def _sample_from_space(space: list, current_value=None):
  """从离散搜索空间中采样，如果提供当前值，则从相邻值中采样"""
  import random
  if current_value is None:
    return random.choice(space)
  # 找到当前值在空间中的位置
  try:
    idx = space.index(current_value)
    # 在相邻位置采样（允许边界）
    if idx == 0:
      return random.choice(space[:2])
    elif idx == len(space) - 1:
      return random.choice(space[-2:])
    else:
      return random.choice(space[idx-1:idx+2])
  except ValueError:
    # 如果当前值不在空间中，随机选择
    return random.choice(space)

def _individual_config_to_key(config: dict) -> tuple:
  """将Individual_config转换为可哈希的key"""
  import json
  return tuple(sorted([
    (k, tuple(sorted(v.items())) if isinstance(v, dict) else v)
    for k, v in config.items()
  ]))

def uniform_crossover(parent1: dict, parent2: dict) -> dict:
  """均匀交叉：每个因子50%概率来自父/母"""
  import random
  return {
    factor: parent1[factor] if random.random() < 0.5 else parent2[factor]
    for factor in parent1.keys()
  }

def mutate(individual: dict, mutation_rate: float = 0.2, search_space: list = None) -> dict:
  """变异：20%概率，从离散搜索空间中采样新值"""
  import random
  mutated = individual.copy()
  for factor in mutated.keys():
    if random.random() < mutation_rate:
      if search_space:
        mutated[factor] = _sample_from_space(search_space, current_value=mutated[factor])
      else:
        # 如果没有提供搜索空间，使用默认的weights空间
        mutated[factor] = _sample_from_space(GA_SEARCH_SPACES['weights'], current_value=mutated[factor])
  return mutated

# GA 状态（全局）
_ga_state = {
  'population': [],  # 当前种群
  'hall_of_fame': [],  # 历史最优池
  'fitness_cache': {},  # 适应度缓存
}

def ga_optimizer(results, population_size: int = 24, hall_of_fame_size: int = 24) -> list[dict]:
  """
  GA 优化器：根据回测结果生成下一代Individual_config

  Args:
    results: 回测结果，每个元素包含 {'individual_config': dict, 'total_return': float, ...}
    population_size: 种群大小
    hall_of_fame_size: 历史最优池大小

  Returns:
    下一代Individual_config列表
  """
  import random

  results_list = list(results) if not isinstance(results, list) else results

  # 1. 更新适应度缓存
  for r in results_list:
    key = _individual_config_to_key(r['individual_config'])
    _ga_state['fitness_cache'][key] = r['total_return']

  testback_logger.info(f"GA 本轮有效结果: {len(results_list)} 个")

  # 2. 初始化种群
  if not _ga_state['population']:
    _ga_state['population'] = [r['individual_config'] for r in results_list][:population_size]
    testback_logger.info(f"GA 初始化种群: {len(_ga_state['population'])} 个个体")

  # 3. 选择存活个体（父代）
  def get_fitness(config):
    key = _individual_config_to_key(config)
    return _ga_state['fitness_cache'][key]

  population_with_fitness = [(ind, get_fitness(ind)) for ind in _ga_state['population']]
  population_with_fitness.sort(key=lambda x: x[1], reverse=True)
  parents = [ind for ind, _ in population_with_fitness[:population_size]]

  # 4. 更新历史最优池
  all_individuals = _ga_state['hall_of_fame'] + parents
  unique_dict = {}
  for ind in all_individuals:
    key = _individual_config_to_key(ind)
    fitness = get_fitness(ind)
    if key not in unique_dict or fitness > unique_dict[key][1]:
      unique_dict[key] = (ind, fitness)
  sorted_hof = sorted(unique_dict.values(), key=lambda x: x[1], reverse=True)
  _ga_state['hall_of_fame'] = [ind for ind, _ in sorted_hof[:hall_of_fame_size]]

  # 5. 生成子代（交叉 + 变异）
  def crossover_config(p1, p2):
    child_weights = uniform_crossover(p1['weights'], p2['weights'])
    child_temperatures = uniform_crossover(p1['temperatures'], p2['temperatures'])
    child_buy_n = p1['buy_n'] if random.random() < 0.5 else p2['buy_n']
    child_sell_m = p1['sell_m'] if random.random() < 0.5 else p2['sell_m']
    # 确保 sell_m >= buy_n，如果不满足则从搜索空间中选择 >= buy_n 的最小值
    if child_sell_m < child_buy_n:
      valid_sell_m = [m for m in GA_SEARCH_SPACES['sell_m'] if m >= child_buy_n]
      child_sell_m = valid_sell_m[0] if valid_sell_m else child_buy_n
    return {'weights': child_weights, 'buy_n': child_buy_n, 'sell_m': child_sell_m, 'temperatures': child_temperatures}

  def mutate_config(config, mutation_rate: float = 0.2):
    # weights 和 temperatures 从离散空间采样
    mutated_weights = mutate(config['weights'], mutation_rate, GA_SEARCH_SPACES['weights'])
    mutated_temperatures = mutate(config['temperatures'], mutation_rate, GA_SEARCH_SPACES['temperatures'])
    
    mutated_buy_n = config['buy_n']
    mutated_sell_m = config['sell_m']
    
    # buy_n 从离散空间采样
    if random.random() < mutation_rate:
      mutated_buy_n = _sample_from_space(GA_SEARCH_SPACES['buy_n'], current_value=mutated_buy_n)
    
    # sell_m 从离散空间采样，但必须 >= buy_n
    if random.random() < mutation_rate:
      valid_sell_m = [m for m in GA_SEARCH_SPACES['sell_m'] if m >= mutated_buy_n]
      if valid_sell_m:
        mutated_sell_m = _sample_from_space(valid_sell_m, current_value=mutated_sell_m)
      else:
        mutated_sell_m = mutated_buy_n
    
    # 最终检查确保 sell_m >= buy_n
    if mutated_sell_m < mutated_buy_n:
      valid_sell_m = [m for m in GA_SEARCH_SPACES['sell_m'] if m >= mutated_buy_n]
      mutated_sell_m = valid_sell_m[0] if valid_sell_m else mutated_buy_n
    
    return {'weights': mutated_weights, 'buy_n': mutated_buy_n, 'sell_m': mutated_sell_m, 'temperatures': mutated_temperatures}

  children = []
  while len(children) < population_size:
    p1, p2 = random.sample(parents, 2)
    child = crossover_config(p1, p2)
    child = mutate_config(child, mutation_rate=0.2)
    children.append(child)

  # 6. 更新当前种群
  _ga_state['population'] = parents + children

  # 7. 输出当前最优
  best = _ga_state['hall_of_fame'][0]
  best_fitness = get_fitness(best)
  testback_logger.info(f"GA 当前最优收益率: {best_fitness:.2f}%, buy_n={best['buy_n']}, sell_m={best['sell_m']}")

  # 8. 返回下一代需要评估的Individual_config：父代 + 子代 + 历史最优
  next_configs = parents + children + _ga_state['hall_of_fame']
  return next_configs[:3 * population_size]

# 全局缓存
testback_cache = SharedMemoryCache[list[TopN]]('testback_cache', compress_level=6)

def _wrap_process_worker(individual_config: dict, mem_offset: int, mem_count: int):
  """ 独立进程计算最终收益 - 从共享内存读取数据

  Args:
    individual_config: Individual_config JSON，包含 weights, buy_n, sell_m, temperatures
    mem_offset: 共享内存偏移
    mem_count: 共享内存数量
  """
  import time
  # 生成 worker 标识
  import os
  worker_id = f"[Worker-{os.getpid()}]"
  weights = individual_config['weights']
  buy_n = individual_config['buy_n']
  sell_m = individual_config['sell_m']
  temperatures = individual_config['temperatures']

  testback_logger.info(f"{worker_id} 🚀 开始回测 | offset={mem_offset}, count={mem_count}, config={individual_config}")

  try:
    # 延迟导入：每个 worker 只导入自己需要的模块
    t_import = time.time()
    from xtquant import xtdata
    xtdata.enable_hello = False
    from core.strategies.sizers.sizer import Sizer
    from core.database import get_market_data
    testback_logger.debug(f"{worker_id} ⏱️  导入模块耗时: {time.time() - t_import:.2f}秒")

    # 创建缓存实例并从共享内存读取
    t_load = time.time()
    topn_list_all = testback_cache.get('topn_data')
    testback_logger.debug(f"{worker_id} ⏱️  从共享内存读取数据耗时: {time.time() - t_load:.2f}秒")

    # 检查是否成功写入
    if not topn_list_all or topn_list_all is None or len(topn_list_all) == 0:
      testback_logger.error(f"{worker_id} ❌ 共享内存数据为空，请先调用 testback_cache.put('topn_data', data) 写入数据")
      exit(1)

    # 读取指定范围的 TopN 实例
    topn_list = topn_list_all[mem_offset:mem_offset + mem_count]
    testback_logger.info(
      f"{worker_id} 📊 数据加载完成 | 总数据={len(topn_list_all)}天, "
      f"当前切片=[{mem_offset}:{mem_offset + mem_count}], 实际={len(topn_list)}天, "
      f"周期: {topn_list[0].base_date} ~ {topn_list[-1].base_date}"
    )

    # 创建账户模拟器
    account = StockAccountMocker(
      cash=500_000.0,  # 初始资金50万元
      commission=2 / 1000,  # 千2交易费率
      min_commission=5.0,  # 最小5元交易费
    )

    # 遍历每个交易日
    for topn in topn_list:
      trade_date = topn.base_date
      trade_datetime = datetime.combine(trade_date, datetime.min.time())

      # 获取当日 Top sell_m 只股票用于判断卖出
      sell_m_stocks = topn.get_ordered_stocks(
        n=sell_m,
        weights=weights,
        temperatures=temperatures,
        norm_method='rank'
      )

      # 获取当日 Top buy_n 只股票用于买入
      buy_n_stocks = topn.get_ordered_stocks(
        n=buy_n,
        weights=weights,
        temperatures=temperatures,
        norm_method='rank'
      )

      all_trade_stocks = list(set(sell_m_stocks + buy_n_stocks))
      if not all_trade_stocks:
        continue

      # 获取股票价格（需要获取所有可能交易的股票价格）
      prices: dict[str, float] = {}
      for stock in all_trade_stocks:
        try:
          # 将 date 转换为 datetime
          data = get_market_data_from_cache(stock, 1, trade_datetime)
          if data is not None and len(data) > 0:
            prices[stock] = float(data.iloc[-1]['close'])
        except Exception as e:
          testback_logger.warning(f"{stock} 价格获取失败: {e}")

      # 卖出不在Top sell_m中的股票
      for stock in set([p for p in account.positions.keys()]) - set(sell_m_stocks):
        if stock not in prices:
          continue  # 跳过没有价格的股票
        account.clear_stock(
          code=stock,
          price=prices[stock],
          clear_date=trade_date,
        )

      # 计算仓位分配（基于buy_n只股票，使用总资产）
      # 只选择有价格的股票
      buy_n_stocks_with_price = [s for s in buy_n_stocks if s in prices]
      if not buy_n_stocks_with_price:
        continue
      allocations = Sizer.allocate(
        [(s, prices[s]) for s in buy_n_stocks_with_price],
        account.calc_assets(trade_datetime)['total_asset'],
      )

      # 买入新股票（Top buy_n中不在当前持仓的）
      for stock, shares in allocations.items():
        if shares <= 0:
          continue

        if stock in account.positions:
          # 已持仓，跳过（不加仓）
          continue

        try:
          account.buy_stock(
            code=stock,
            volume=shares,
            price=prices[stock],
            buy_date=trade_date
          )
        except Exception as e:
          testback_logger.warning(f"买入失败 {stock}: {e}")

    # 计算最终收益
    final_datetime = datetime.combine(topn_list[-1].base_date, datetime.min.time())
    final_assets = account.calc_assets(final_datetime)
    total_return = (final_assets['total_asset'] - account.init_cash) / account.init_cash * 100

    return {
      'individual_config': individual_config,
      'init_cash': account.init_cash,
      'final_cash': final_assets['cash'],
      'final_market_value': final_assets['market_value'],
      'final_total_asset': final_assets['total_asset'],
      'total_return': total_return,
      'cleared_positions_count': len(account.cleared_positions),
      'current_positions_count': len(account.positions),
    }

  except Exception as e:
    testback_logger.error(f"回测时出错: {e}")
    import traceback
    testback_logger.error(traceback.format_exc())
    return None

if __name__ == "__main__":
  import random
  from core.database import get_all_stock_code_list
  from utils.stock.time import get_trading_date_span

  ts = datetime.now()
  all_stocks = get_all_stock_code_list()

  # 预加载股票详情到共享内存缓存
  init_stock_detail_cache(all_stocks)
  # 初始化并预加载数据到共享内存
  init_full_data(all_stocks, '1d')

  backtest_datetime_list = [
    datetime.combine(d, datetime.min.time())
    for d in get_trading_date_span(date(2020, 6, 30), date(2024, 12, 31))]

  # 多进程获取 TopN 实例 预加载不包含weights
  topn_worker_count = min(os.cpu_count(), len(backtest_datetime_list))
  with parallel_backend('loky', n_jobs=topn_worker_count):
    parallel_pool = Parallel(
      return_as='generator',  # 结果以生成器形式返回
      n_jobs=topn_worker_count,
      prefer='processes',
      batch_size=1,
      verbose=0,
    )
    topNs = list(tqdm(
      parallel_pool(delayed(TopN)(all_stocks, d) for d in backtest_datetime_list),
      total=len(backtest_datetime_list),
      maxinterval=30,
      desc=f"预热 TopN 股票：{len(all_stocks)} 只股票在 {backtest_datetime_list[0]} ~ {backtest_datetime_list[-1]}，{len(backtest_datetime_list)}天"
    ))

  # 清理预加载的全局缓存，释放内存
  cleanup_shared_cache()

  # 创建共享内存缓存并存入数据
  ordered_topNs = sorted(topNs, key=lambda x: x.base_date)
  testback_cache.put('topn_data', ordered_topNs)
  testback_logger.debug(f"已将 {len(ordered_topNs)} 天的 TopN 实例存入共享内存缓存")

  # 可选：从文件加载Individual_config（用于单次回测）
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--individual-config', type=str, default=None, help='Individual_config JSON文件路径')
  parser.add_argument('--period-span', type=int, default=30, help='回测周期天数（默认30天）')
  args = parser.parse_args()

  if args.individual_config:
    import json

    with open(args.individual_config, 'r', encoding='utf-8') as f:
      config_data = json.load(f)
    individual_config = config_data['individual_config']
    testback_logger.info(f"从文件加载Individual_config: {args.individual_config}")
    testback_logger.info(f"使用配置进行单次回测: buy_n={individual_config['buy_n']}, sell_m={individual_config['sell_m']}")

    data_idx = 0
    period_span = args.period_span
    result = _wrap_process_worker(individual_config, data_idx, period_span)
    testback_logger.info(f"回测完成: 总收益={result['total_return']:.2f}%")
    exit(0)

  # GA优化模式
  GENERATIONS = 1000
  # 种群大小根据 CPU 核心数向下取双数倍
  cpu_count = os.cpu_count() or 4
  POPULATIONS = (cpu_count // 2) * 2  # 向下取双数倍，如 12->10, 8->8, 6->6, 4->4
  GA_PERIOD_SPAN = 30
  ga_worker_count = min(cpu_count, GENERATIONS)
  data_idx = 0
  ALL_FACTOR_NAMES = [
    'SmallCap',
    'WMACross',
  ]

  initial_weights = None

  # 生成初始Individual_config列表
  def generate_initial_config(index: int) -> dict:
    import random
    # 从离散搜索空间采样
    buy_n = random.choice(GA_SEARCH_SPACES['buy_n'])
    # sell_m 必须 >= buy_n
    valid_sell_m = [m for m in GA_SEARCH_SPACES['sell_m'] if m >= buy_n]
    sell_m = random.choice(valid_sell_m) if valid_sell_m else buy_n
    # temperatures 和 weights 从离散空间采样
    temperatures = {factor: random.choice(GA_SEARCH_SPACES['temperatures']) for factor in ALL_FACTOR_NAMES}
    weights = {factor: random.choice(GA_SEARCH_SPACES['weights']) for factor in ALL_FACTOR_NAMES}
    return {
      'weights': weights,
      'buy_n': buy_n,
      'sell_m': sell_m,
      'temperatures': temperatures
    }

  new_params = []
  for i in range(3 * POPULATIONS):
    config = generate_initial_config(i)
    new_params.append([config, data_idx, GA_PERIOD_SPAN])
  testback_logger.debug(f"开始回测：{GENERATIONS}个任务，{ga_worker_count}个进程，共{len(all_stocks)}只股票")
  with parallel_backend('loky', n_jobs=ga_worker_count):
    parallel_pool = Parallel(
      return_as='generator',  # 结果以生成器形式返回
      n_jobs=ga_worker_count,
      prefer='processes',
      batch_size=1,
      verbose=0,
    )
  all_results = []
  generation_results = []  # 按代保存的结果

  # 创建结果目录
  results_dir = Path('results')
  results_dir.mkdir(exist_ok=True)
  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  ga_result_dir = results_dir / f'ga_{timestamp}'
  ga_result_dir.mkdir(exist_ok=True)

  for generation in range(GENERATIONS):
    testback_logger.info(f"\n{'=' * 60}")
    testback_logger.info(f"GA 第 {generation + 1}/{GENERATIONS} 代")
    testback_logger.info(f"{'=' * 60}")

    results = parallel_pool(
      delayed(_wrap_process_worker)(*args)
      for args in new_params
    )
    # 将生成器转换为列表
    results_list = list(results)

    # 为每个结果添加代数信息和适应度
    for result in results_list:
      result['generation'] = generation
      result['fitness'] = result['total_return']

    all_results.extend(results_list)

    # 保存该代的统计信息
    fitnesses = [ind['fitness'] for ind in results_list]
    generation_stats = {
      'generation': generation,
      'generation_time': 0,
      'population_size': len(results_list),
      'max_fitness': max(fitnesses),
      'mean_fitness': sum(fitnesses) / len(fitnesses),
      'min_fitness': min(fitnesses),
      'all_individuals': results_list
    }
    generation_results.append(generation_stats)

    # 每代保存结果（防止中断丢失数据）
    import pickle
    import json
    with open(ga_result_dir / 'generation_results.pkl', 'wb') as f:
      pickle.dump(generation_results, f)
    # 保存当前最优
    best_in_gen = max(results_list, key=lambda x: x['fitness'])
    best_result = {
      'individual_config': best_in_gen['individual_config'],
      'fitness': best_in_gen['fitness'],
      'generation': generation + 1,
      'generation_time': generation_stats['generation_time'],
      'population_size': len(results_list),
      'all_factor_names': ALL_FACTOR_NAMES
    }
    with open(ga_result_dir / 'best_individual_config.json', 'w', encoding='utf-8') as f:
      json.dump(best_result, f, indent=2, ensure_ascii=False)

    next_configs = ga_optimizer(results_list)
    data_idx = random.randrange(0, len(ordered_topNs) - GA_PERIOD_SPAN)
    new_params = [[config, data_idx, GA_PERIOD_SPAN] for config in next_configs]

  # 保存所有个体的详细信息（Individual_config列表）
  all_individual_configs = []
  for gen_stat in generation_results:
    for ind in gen_stat['all_individuals']:
      all_individual_configs.append({
        'generation': ind['generation'],
        'individual_config': ind['individual_config'],
        'fitness': ind['fitness'],
        'total_return': ind['total_return'],
        'init_cash': ind['init_cash'],
        'final_cash': ind['final_cash'],
        'final_market_value': ind['final_market_value'],
        'final_total_asset': ind['final_total_asset'],
        'cleared_positions_count': ind['cleared_positions_count'],
        'current_positions_count': ind['current_positions_count'],
      })

  # 保存为JSON格式
  import json

  with open(ga_result_dir / 'all_individuals.json', 'w', encoding='utf-8') as f:
    json.dump(all_individual_configs, f, indent=2, ensure_ascii=False)
  testback_logger.info(f"已保存所有个体配置: {ga_result_dir / 'all_individuals.json'}")

  # 最终更新最优配置（hall_of_fame）
  best_config = _ga_state['hall_of_fame'][0]
  key = _individual_config_to_key(best_config)
  best_fitness = _ga_state['fitness_cache'][key]

  best_result = {
    'individual_config': best_config,
    'fitness': best_fitness,
    'generation': GENERATIONS,
    'population_size': POPULATIONS,
    'all_factor_names': ALL_FACTOR_NAMES
  }
  with open(ga_result_dir / 'best_individual_config.json', 'w', encoding='utf-8') as f:
    json.dump(best_result, f, indent=2, ensure_ascii=False)

  testback_logger.info(f"\n最优参数已保存:")
  testback_logger.info(f"  - {ga_result_dir / 'best_individual_config.json'}")
  testback_logger.info(f"最优收益率: {best_fitness:.2f}%")
  testback_logger.info(f"最优buy_n: {best_config['buy_n']}, sell_m: {best_config['sell_m']}")

  testback_logger.info(f"\n{'=' * 60}")
  testback_logger.info(f"回测执行完成")

  returns = [r['total_return'] for r in all_results]
  testback_logger.info(f"\n统计信息:")
  testback_logger.info(f"  总回测次数: {len(all_results)}")
  testback_logger.info(f"  平均收益率: {sum(returns) / len(returns):.2f}%")
  testback_logger.info(f"  最大收益率: {max(returns):.2f}%")
  testback_logger.info(f"  最小收益率: {min(returns):.2f}%")
  testback_logger.info(f"  正收益策略: {len([r for r in returns if r > 0])} 个")
  testback_logger.info(f"  负收益策略: {len([r for r in returns if r < 0])} 个")
  testback_logger.info(f"{'=' * 60}")
  te = datetime.now()
  testback_logger.info(f"总耗时: {(te - ts).total_seconds():.2f} 秒")
