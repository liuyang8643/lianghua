"""
强化学习环境使用示例

展示如何在回测场景中使用 RLEnv，包括：
1. 简单的随机策略
2. 基于规则的策略
3. 与多进程回测集成
"""
import os
import random
from datetime import date, datetime, timedelta
from typing import Dict, Any

from testback.logger import testback_logger
from testback.rl_env import RLEnv, create_rl_env_from_cache
from core.strategies.sizers.sizer import Sizer

# ==================== 策略示例 ====================

class RandomAgent:
  """随机策略：随机买卖"""

  def __init__(self, max_positions: int = 10):
    self.max_positions = max_positions

  def select_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
    """
    随机选择动作

    Args:
        observation: 环境观测

    Returns:
        动作字典
    """
    candidate_stocks = observation['candidate_stocks']
    current_positions = observation['current_positions']
    prices = observation['prices']
    cash = observation['cash']

    action = {
      'stocks_to_buy': [],
      'stocks_to_sell': []
    }

    # 随机卖出一些持仓
    if current_positions:
      stocks_to_sell = random.sample(
        list(current_positions.keys()),
        k=random.randint(0, len(current_positions))
      )
      for stock in stocks_to_sell:
        action['stocks_to_sell'].append(
          (stock, current_positions[stock])
        )

    # 随机买入一些候选股票
    available_stocks = [
      s for s in candidate_stocks
      if s not in current_positions and s in prices
    ]

    if available_stocks and cash > 10000:
      num_to_buy = min(
        random.randint(1, 3),
        self.max_positions - len(current_positions),
        len(available_stocks)
      )

      if num_to_buy > 0:
        stocks_to_buy = random.sample(available_stocks, k=num_to_buy)
        buy_amount = cash / num_to_buy

        for stock in stocks_to_buy:
          price = prices[stock]
          shares = int(buy_amount / price / 100) * 100  # 整手
          if shares >= 100:
            action['stocks_to_buy'].append((stock, shares))

    return action

class TopNAgent:
  """Top-N策略：买入得分最高的N只股票"""

  def __init__(self, n: int = 10):
    self.n = n

  def select_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
    """
    选择Top N股票买入，卖出不在Top N中的持仓

    Args:
        observation: 环境观测

    Returns:
        动作字典
    """
    candidate_stocks = observation['candidate_stocks'][:self.n]  # 取前N只
    current_positions = observation['current_positions']
    prices = observation['prices']
    cash = observation['cash']

    action = {
      'stocks_to_buy': [],
      'stocks_to_sell': []
    }

    # 卖出不在Top N中的持仓
    target_holdings = set(candidate_stocks)
    stocks_to_sell = set(current_positions.keys()) - target_holdings

    for stock in stocks_to_sell:
      action['stocks_to_sell'].append(
        (stock, current_positions[stock])
      )

    # 买入Top N中未持仓的股票
    stocks_to_buy = [
      s for s in candidate_stocks
      if s not in current_positions and s in prices
    ]

    if stocks_to_buy and cash > 10000:
      # 使用Sizer分配资金
      allocations = Sizer.allocate(
        stocks=stocks_to_buy,
        total_capital=cash,
        prices={s: prices[s] for s in stocks_to_buy}
      )

      for stock, shares in allocations.items():
        if shares >= 100:
          action['stocks_to_buy'].append((stock, shares))

    return action

class HoldAgent:
  """持有策略：首次买入后一直持有"""

  def __init__(self, n: int = 10):
    self.n = n
    self.initialized = False

  def select_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
    """
    首次买入Top N，之后一直持有

    Args:
        observation: 环境观测

    Returns:
        动作字典
    """
    action = {
      'stocks_to_buy': [],
      'stocks_to_sell': []
    }

    # 如果已初始化，什么都不做
    if self.initialized:
      return action

    # 首次买入
    candidate_stocks = observation['candidate_stocks'][:self.n]
    prices = observation['prices']
    cash = observation['cash']

    stocks_to_buy = [s for s in candidate_stocks if s in prices]

    if stocks_to_buy and cash > 10000:
      allocations = Sizer.allocate(
        stocks=stocks_to_buy,
        total_capital=cash,
        prices={s: prices[s] for s in stocks_to_buy}
      )

      for stock, shares in allocations.items():
        if shares >= 100:
          action['stocks_to_buy'].append((stock, shares))

      self.initialized = True

    return action

# ==================== 训练循环示例 ====================

def run_episode_with_agent(env: RLEnv, agent: Any) -> Dict[str, Any]:
  """
  运行一个完整的回测回合

  Args:
      env: RL环境
      agent: 策略（需实现 select_action 方法）

  Returns:
      回合统计信息
  """
  # 重置环境
  state = env.reset()
  total_reward = 0.0

  testback_logger.info(f"开始回测，共{env.max_steps}个交易日")

  # 运行回合
  while not env.done():
    # 观测状态
    obs = env._get_observation()

    # 智能体决策
    action = agent.select_action(obs)

    # 执行动作
    next_state, reward, done, info = env.step(action)

    # 累计奖励
    total_reward += reward

    # 日志
    testback_logger.debug(
      f"[{info['date']}] "
      f"收益: {reward:.2f}%, "
      f"累计: {info['cumulative_return']:.2f}%, "
      f"持仓: {info['positions_count']}只, "
      f"资产: {info['total_asset']:,.0f}"
    )

    # 更新状态
    state = next_state

  # 获取回合统计
  summary = env.get_episode_summary()

  # 输出结果（根据是否使用alpha调整输出）
  if env.use_alpha_reward:
    testback_logger.info(
      f"回测完成：总收益率 {summary['total_return']:.2f}%, "
      f"基准收益率 {summary['baseline_return']:.2f}%, "
      f"Alpha {summary['alpha']:.2f}%, "
      f"信息比率 {summary['info_ratio']:.2f}"
    )
  else:
    testback_logger.info(
      f"回测完成：总收益率 {summary['total_return']:.2f}%, "
      f"夏普比率 {summary['sharpe_ratio']:.2f}"
    )

  return summary

# ==================== 多进程集成 ====================

def rl_worker(weights: dict[str, float], agent_type: str = 'topn', rank_n: int = 20):
  """
  用于多进程的RL worker（可以直接替换 _wrap_process_worker）

  Args:
      weights: 因子权重
      agent_type: 策略类型 ('random', 'topn', 'hold')
      rank_n: 选股数量

  Returns:
      回测结果
  """
  try:
    # 延迟导入
    from xtquant import xtdata
    xtdata.enable_hello = False

    # 从共享内存创建环境
    env = create_rl_env_from_cache(
      weights=weights,
      rank_n=rank_n,
      init_cash=500_000.0,
      commission=2 / 1000,
      min_commission=5.0,
    )

    # 选择策略
    if agent_type == 'random':
      agent = RandomAgent(max_positions=10)
    elif agent_type == 'topn':
      agent = TopNAgent(n=10)
    elif agent_type == 'hold':
      agent = HoldAgent(n=10)
    else:
      raise ValueError(f"未知的策略类型: {agent_type}")

    # 运行回测
    summary = run_episode_with_agent(env, agent)

    # 添加权重信息
    summary['weights'] = weights
    summary['agent_type'] = agent_type

    return summary

  except Exception as e:
    testback_logger.error(f"RL worker 出错: {e}")
    import traceback
    testback_logger.error(traceback.format_exc())
    return None

# ==================== 单进程测试 ====================

if __name__ == "__main__":
  """单进程测试示例"""
  from utils.parallel import batch_run_threads
  from core.strategies import TopN
  from core.database import allow_buy_stock_code_list
  from utils.stock.time import get_trading_date_span
  from utils.shared_memory import SharedMemoryCache

  # 准备数据
  testback_logger.info("准备测试数据...")

  all_stocks = allow_buy_stock_code_list()[:10]  # 只用10只股票
  today = date.today()
  start_date = today - timedelta(days=10)
  back_dates = get_trading_date_span(start_date, today)[-5:]  # 最近5个交易日

  testback_logger.info(f"股票数量: {len(all_stocks)}")
  testback_logger.info(f"回测日期: {[d.strftime('%Y-%m-%d') for d in back_dates]}")

  # 获取TopN数据
  testback_logger.info("计算因子得分...")
  topNs = batch_run_threads(
    func=TopN,
    args_list=[[all_stocks, datetime.combine(d, datetime.min.time())] for d in back_dates],
    max_workers=4,
  )

  # 写入共享内存（用于测试多进程场景）
  testback_cache = SharedMemoryCache('testback_cache', compress_level=6)
  testback_cache.put('topn_data', topNs)

  # 创建环境（启用alpha奖励）
  testback_logger.info("创建RL环境（使用超额收益alpha作为奖励）...")
  env = RLEnv(
    topn_list=topNs,
    init_cash=500_000.0,
    rank_n=20,
    weights={'MACD': 1.0, 'BBI': 1.0, 'CCI': 1.0},
    use_alpha_reward=True,  # 启用alpha奖励
  )

  # 测试不同策略
  agents = {
    '随机策略': RandomAgent(max_positions=10),
    'TopN策略': TopNAgent(n=10),
    '持有策略': HoldAgent(n=10),
  }

  results = {}
  for name, agent in agents.items():
    testback_logger.info(f"\n{'=' * 60}")
    testback_logger.info(f"测试策略: {name}")
    testback_logger.info(f"{'=' * 60}")

    summary = run_episode_with_agent(env, agent)
    results[name] = summary

    testback_logger.info(f"\n{name} 回测结果:")
    testback_logger.info(f"  总收益率: {summary['total_return']:.2f}%")
    if env.use_alpha_reward:
      testback_logger.info(f"  基准收益: {summary['baseline_return']:.2f}%")
      testback_logger.info(f"  超额收益(Alpha): {summary['alpha']:.2f}%")
      testback_logger.info(f"  信息比率: {summary['info_ratio']:.2f}")
    testback_logger.info(f"  夏普比率: {summary['sharpe_ratio']:.2f}")
    testback_logger.info(f"  最终资产: {summary['final_total_asset']:,.0f}")
    testback_logger.info(f"  平均奖励: {summary['avg_reward']:.4f}%")

  # 对比结果
  testback_logger.info(f"\n{'=' * 60}")
  testback_logger.info("策略对比")
  testback_logger.info(f"{'=' * 60}")

  sorted_results = sorted(
    results.items(),
    key=lambda x: x[1].get('alpha', x[1]['total_return']),  # 按alpha排序
    reverse=True
  )

  for i, (name, summary) in enumerate(sorted_results, 1):
    if env.use_alpha_reward:
      testback_logger.info(
        f"#{i} {name}: Alpha {summary['alpha']:.2f}% "
        f"(总收益: {summary['total_return']:.2f}%, "
        f"基准: {summary['baseline_return']:.2f}%)"
      )
    else:
      testback_logger.info(
        f"#{i} {name}: {summary['total_return']:.2f}% "
        f"(夏普: {summary['sharpe_ratio']:.2f})"
      )
