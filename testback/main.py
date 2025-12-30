import os

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

def _dict_to_key(d: dict) -> tuple:
  """将字典转换为可哈希的key"""
  return tuple(sorted(d.items()))

def uniform_crossover(parent1: dict, parent2: dict) -> dict:
  """均匀交叉：每个因子50%概率来自父/母"""
  import random
  return {
    factor: parent1[factor] if random.random() < 0.5 else parent2[factor]
    for factor in parent1.keys()
  }

def mutate(individual: dict, mutation_rate: float = 0.2) -> dict:
  """变异：20%概率，幅度为当前值的±(10%-30%)，限制在[-1,1]"""
  import random
  mutated = individual.copy()
  for factor in mutated.keys():
    if random.random() < mutation_rate:
      mutation_amplitude = random.uniform(0.1, 0.3)
      delta = mutated[factor] * mutation_amplitude * random.choice([-1, 1])
      mutated[factor] = max(-1, min(1, mutated[factor] + delta))
  return mutated

# GA 状态（全局）
_ga_state = {
  'population': [],       # 当前种群
  'hall_of_fame': [],     # 历史最优池
  'fitness_cache': {},    # 适应度缓存
}

def ga_optimizer(results, population_size: int = 24, hall_of_fame_size: int = 24) -> list[dict]:
  """
  GA 优化器：根据回测结果生成下一代权重

  Args:
    results: 回测结果（列表或生成器），每个元素包含 {'weights': dict, 'total_return': float, ...}
    population_size: 种群大小
    hall_of_fame_size: 历史最优池大小

  Returns:
    下一代权重列表，长度为 3 * population_size（父代 + 子代 + 历史最优）
  """
  import random

  # 将生成器转换为列表
  results_list = list(results) if not isinstance(results, list) else results

  # 1. 更新适应度缓存
  valid_results = []
  for r in results_list:
    if r is not None and 'weights' in r and 'total_return' in r:
      key = _dict_to_key(r['weights'])
      _ga_state['fitness_cache'][key] = r['total_return']
      valid_results.append(r)

  testback_logger.info(f"GA 本轮有效结果: {len(valid_results)} 个")

  # 2. 如果是第一代，初始化种群
  if not _ga_state['population']:
    _ga_state['population'] = [r['weights'] for r in valid_results][:population_size]
    testback_logger.info(f"GA 初始化种群: {len(_ga_state['population'])} 个个体")

  # 检查种群是否足够
  if len(_ga_state['population']) < 2:
    testback_logger.error(f"GA 种群不足: {len(_ga_state['population'])} 个，需要至少2个")
    # 返回当前已有的权重，让下一轮继续
    return [r['weights'] for r in valid_results] if valid_results else []

  # 3. 选择存活个体（父代）- 从当前种群中选择最优的 population_size 个
  population_with_fitness = [
    (ind, _ga_state['fitness_cache'].get(_dict_to_key(ind), float('-inf')))
    for ind in _ga_state['population']
  ]
  population_with_fitness.sort(key=lambda x: x[1], reverse=True)
  parents = [ind for ind, _ in population_with_fitness[:population_size]]

  # 确保 parents 至少有2个
  if len(parents) < 2:
    testback_logger.warning(f"GA parents 不足: {len(parents)} 个")
    return [r['weights'] for r in valid_results] if valid_results else []

  # 4. 更新历史最优池
  all_individuals = _ga_state['hall_of_fame'] + parents
  unique_dict = {}
  for ind in all_individuals:
    key = _dict_to_key(ind)
    fitness = _ga_state['fitness_cache'].get(key, float('-inf'))
    if key not in unique_dict or fitness > unique_dict[key][1]:
      unique_dict[key] = (ind, fitness)
  sorted_hof = sorted(unique_dict.values(), key=lambda x: x[1], reverse=True)
  _ga_state['hall_of_fame'] = [ind for ind, _ in sorted_hof[:hall_of_fame_size]]

  # 5. 生成子代（交叉 + 变异）
  children = []
  while len(children) < population_size:
    p1, p2 = random.sample(parents, 2)
    child = uniform_crossover(p1, p2)
    child = mutate(child, mutation_rate=0.2)
    children.append(child)

  # 6. 更新当前种群（用于下一轮选择）
  _ga_state['population'] = parents + children

  # 7. 输出当前最优
  if _ga_state['hall_of_fame']:
    best = _ga_state['hall_of_fame'][0]
    best_fitness = _ga_state['fitness_cache'].get(_dict_to_key(best), float('-inf'))
    testback_logger.info(f"GA 当前最优收益率: {best_fitness:.2f}%")

  # 8. 返回下一代需要评估的权重：父代 + 子代 + 历史最优
  next_weights = parents + children + _ga_state['hall_of_fame']
  return next_weights[:3 * population_size]

# 全局缓存
testback_cache = SharedMemoryCache[list[TopN]]('testback_cache', compress_level=6)

def _wrap_process_worker(weights: dict[str, float], mem_offset: int, mem_count: int):
  """ 独立进程计算最终收益 - 从共享内存读取数据

  Args:
    weights: 因子权重
  """
  import time
  t_start = time.time()

  rank_n = 20
  use_rl_env = False

  # 生成 worker 标识
  import os
  worker_id = f"[Worker-{os.getpid()}]"

  testback_logger.info(f"{worker_id} 🚀 开始回测 | offset={mem_offset}, count={mem_count}, weights={weights}")

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

    # === 可选：使用RL环境 ===
    if use_rl_env:
      from testback.rl_env import RLEnv
      from testback.rl_example import TopNAgent, run_episode_with_agent

      # 创建RL环境（使用alpha奖励）
      env = RLEnv(
        topn_list=topn_list,
        init_cash=500_000.0,
        rank_n=rank_n,
        weights=weights,
        commission=2 / 1000,
        min_commission=5.0,
        use_alpha_reward=True,  # 启用超额收益作为奖励
      )

      # 使用TopN策略
      agent = TopNAgent(n=10)

      # 运行回测
      summary = run_episode_with_agent(env, agent)
      summary['weights'] = weights

      return summary

    # === 原有的逐日回测逻辑 ===
    # 创建账户模拟器
    account = StockAccountMocker(
      cash=500_000.0,  # 初始资金50万元
      commission=2 / 1000,  # 千2交易费率
      min_commission=5.0,  # 最小5元交易费
    )

    # 记录上一日持仓
    prev_holdings = set()

    # 遍历每个交易日
    for topn in topn_list:
      trade_date = topn.base_date

      # 1. 获取当日 Top N 股票（使用传入的权重）
      top_stocks = topn.get_ordered_stocks(
        n=rank_n,
        weights=weights,
        temperatures={k: 1.0 for k in weights.keys()},  # 使用默认温度
        norm_method='rank'
      )

      if not top_stocks:
        continue

      # 2. 获取股票价格
      trade_datetime = datetime.combine(trade_date, datetime.min.time())
      prices = {}
      for stock in top_stocks:
        try:
          # 将 date 转换为 datetime
          data = get_market_data_from_cache(stock, 1, trade_datetime)
          if data is not None and len(data) > 0:
            prices[stock] = float(data.iloc[-1]['close'])
        except Exception as e:
          testback_logger.warning(f"{stock} 价格获取失败: {e}")

      if not prices:
        continue

      # 3. 计算仓位分配
      target_holdings = set(top_stocks)
      allocations = Sizer.allocate(
        stocks=top_stocks,
        total_capital=account.calc_assets(trade_datetime).total_asset,
        prices=prices
      )

      # 4. 卖出不在目标持仓中的股票
      stocks_to_sell = prev_holdings - target_holdings
      for stock in stocks_to_sell:
        pos = account.get_position(stock)
        if pos and pos['volume'] > 0:
          try:
            sell_price = prices.get(stock)
            if not sell_price:
              # 如果没有价格，使用持仓均价
              sell_price = pos['avg_price']
            account.sell_stock(
              code=stock,
              volume=pos['volume'],
              price=sell_price,
              sell_date=trade_date,
              clear_reason='调仓'
            )
          except Exception as e:
            testback_logger.warning(f"卖出 {stock} 失败: {e}")

      # 5. 买入新股票
      for stock, shares in allocations.items():
        if shares <= 0:
          continue

        # 检查是否已持仓
        pos = account.get_position(stock)
        if pos:
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
          # 资金不足或其他错误，跳过
          testback_logger.debug(f"买入 {stock} 失败: {e}")

      # 6. 记录当日资产
      trade_datetime = datetime.combine(trade_date, datetime.min.time())
      account.calc_assets(trade_datetime)

      # 更新持仓记录
      prev_holdings = target_holdings

    # 7. 计算最终收益
    final_datetime = datetime.combine(topn_list[-1].base_date, datetime.min.time())
    final_assets = account.calc_assets(final_datetime)
    total_return = (final_assets['total_asset'] - account.init_cash) / account.init_cash * 100

    return {
      'weights': weights,
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
  from core.database import allow_buy_stock_code_list
  from utils.stock.time import get_trading_date_span

  ts = datetime.now()
  all_stocks = allow_buy_stock_code_list()

  # 预加载股票详情到共享内存缓存
  init_stock_detail_cache(all_stocks)
  # 初始化并预加载数据到共享内存
  init_full_data(all_stocks, '1d')

  backtest_datetime_list = [
    datetime.combine(d, datetime.min.time())
    for d in get_trading_date_span(date(2025, 11, 1), date(2025, 12, 15))]

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

  # 任务数量：模拟 GA 一代的评估量（3k，k=24时为72）
  GENERATIONS = 1000  # 对应 population=24 时一代的评估量
  POPULATIONS = 24    # 权重的数量，正常评估一次是24个子代，24个父代，24个历史最优，总计3*N
  GA_PERIOD_SPAN = 25  # 天，模拟每个任务使用不同时间段的数据
  ga_worker_count = min(os.cpu_count(), GENERATIONS)
  data_idx = 0
  # 所有因子名称列表
  ALL_FACTOR_NAMES = [
    'MACD',
    'BBI',
    'CCI',
    'TRIXFactor',
    'MOMFactor',
    'ADXFactor',
    'SmallCap',
    'Unpopular',
  ]

  new_weights = [
    [
      {factor: random.uniform(-1, 1) for factor in ALL_FACTOR_NAMES},
      data_idx,
      GA_PERIOD_SPAN
    ]
    for _ in range(3*POPULATIONS)
  ]
  testback_logger.debug(f"开始回测：{GENERATIONS}个任务，{ga_worker_count}个进程，共{len(all_stocks)}只股票")
  with parallel_backend('loky', n_jobs=ga_worker_count):
    parallel_pool = Parallel(
      return_as='generator',  # 结果以生成器形式返回
      n_jobs=ga_worker_count,
      prefer='processes',
      batch_size=1,
      verbose=0,
    )
    for generation in range(GENERATIONS):
      results=parallel_pool(
        delayed(_wrap_process_worker)(*args)
        for args in new_weights
      )
      #根据results的结果，去计算下一个需要的优化个体
      weights = ga_optimizer(results)
      data_idx = random.randrange(0,len(ordered_topNs)-GA_PERIOD_SPAN)
      new_weights = [
    [
      weights[i],
      data_idx,
      GA_PERIOD_SPAN
    ]
    for i in range(3*POPULATIONS)
  ]

  # 输出回测结果
  testback_logger.info(f"{'=' * 60}")
  testback_logger.info(f"回测执行完成")
  returns = [r['total_return'] for r in [r for r in results if r is not None]]
  testback_logger.info(f"\n统计信息:")
  testback_logger.info(f"  平均收益率: {sum(returns) / len(returns):.2f}%")
  testback_logger.info(f"  最大收益率: {max(returns):.2f}%")
  testback_logger.info(f"  最小收益率: {min(returns):.2f}%")
  testback_logger.info(f"  正收益策略: {len([r for r in returns if r > 0])} 个")
  testback_logger.info(f"  负收益策略: {len([r for r in returns if r < 0])} 个")
  testback_logger.info(f"{'=' * 60}")
  te = datetime.now()
  testback_logger.info(f"总耗时: {(te - ts).total_seconds():.2f} 秒")
