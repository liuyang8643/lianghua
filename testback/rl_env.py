"""
强化学习环境 - 用于量化交易回测

环境说明：
- 状态空间：因子得分、持仓信息、资金状态
- 动作空间：买入/卖出/持有决策
- 奖励函数：超额收益率（alpha）、夏普比率等
"""
from datetime import date, datetime
from typing import Optional, Dict, List, Tuple, Any
import numpy as np

from core import get_market_data_from_cache
from testback.logger import testback_logger
from testback.account import StockAccountMocker
from core.strategies import TopN
from utils.stock.info import get_baseline_data, baseline_stock_code

class RLEnv:
  """强化学习交易环境"""

  def __init__(
      self,
      topn_list: List[TopN],
      init_cash: float = 500_000.0,
      commission: float = 2 / 1000,
      min_commission: float = 5.0,
      rank_n: int = 20,
      weights: Optional[Dict[str, float]] = None,
      temperatures: Optional[Dict[str, float]] = None,
      norm_method: str = 'rank',
      use_alpha_reward: bool = True,
      baseline_code: str = baseline_stock_code
  ):
    """
    初始化RL环境

    Args:
        topn_list: TopN策略实例列表（每个交易日一个）
        init_cash: 初始资金
        commission: 交易费率
        min_commission: 最小交易费
        rank_n: 选股数量
        weights: 因子权重
        temperatures: 因子温度参数
        norm_method: 归一化方法
        use_alpha_reward: 是否使用超额收益（alpha）作为奖励
        baseline_code: 基准指数代码（默认沪深300）
    """
    self.topn_list = topn_list
    self.rank_n = rank_n
    self.weights = weights or {}
    self.use_alpha_reward = use_alpha_reward
    self.baseline_code = baseline_code
    self.temperatures = temperatures or {}
    self.norm_method = norm_method

    # 初始化账户
    self.account = StockAccountMocker(
      cash=init_cash,
      commission=commission,
      min_commission=min_commission
    )

    # 环境状态
    self.current_step = 0
    self.max_steps = len(topn_list)
    self.is_done = False

    # 候选股票池和价格缓存
    self.candidate_stocks: List[str] = []
    self.stock_prices: Dict[str, float] = {}

    # 历史信息
    self.episode_rewards: List[float] = []
    self.episode_actions: List[Dict] = []

    # 基准收益率缓存（用于计算alpha）
    self.baseline_returns: List[float] = []
    self.prev_baseline_price: Optional[float] = None

    # 预加载基准数据
    if self.use_alpha_reward:
      self._preload_baseline_data()

    testback_logger.info(
      f"RL环境初始化完成：{self.max_steps}个交易日，"
      f"初始资金{init_cash:,.0f}元，选股数量{rank_n}，"
      f"奖励模式：{'超额收益(alpha)' if use_alpha_reward else '绝对收益'}"
    )

  def _preload_baseline_data(self):
    """预加载基准数据，计算每日基准收益率"""
    testback_logger.debug(f"预加载基准数据：{self.baseline_code}")

    baseline_prices = []
    for topn in self.topn_list:
      trade_datetime = datetime.combine(topn.base_date, datetime.min.time())
      baseline_data = get_baseline_data(trade_datetime)

      if baseline_data is not None:
        baseline_prices.append(float(baseline_data['close']))
      else:
        testback_logger.warning(f"无法获取 {topn.base_date} 的基准数据")
        baseline_prices.append(None)

    # 计算基准收益率
    for i in range(len(baseline_prices)):
      if i == 0 or baseline_prices[i] is None or baseline_prices[i - 1] is None:
        self.baseline_returns.append(0.0)
      else:
        baseline_return = (
            (baseline_prices[i] - baseline_prices[i - 1])
            / baseline_prices[i - 1] * 100
        )
        self.baseline_returns.append(baseline_return)

    testback_logger.debug(
      f"基准数据加载完成，{len(self.baseline_returns)}个交易日，"
      f"平均收益率：{np.mean(self.baseline_returns):.2f}%"
    )

  def reset(self) -> Dict[str, Any]:
    """
    重置环境到初始状态

    Returns:
        初始观测状态
    """
    # 重置账户
    self.account = StockAccountMocker(
      cash=self.account.init_cash,
      commission=self.account.commission,
      min_commission=self.account.min_commission
    )

    # 重置环境状态
    self.current_step = 0
    self.is_done = False
    self.candidate_stocks = []
    self.stock_prices = {}
    self.episode_rewards = []
    self.episode_actions = []
    self.prev_baseline_price = None

    # 获取第一天的观测
    return self._get_observation()

  def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
    """
    执行一个交易动作

    Args:
        action: 动作字典 {
            'stocks_to_buy': [(code, shares), ...],
            'stocks_to_sell': [(code, shares), ...]
        }

    Returns:
        (observation, reward, done, info)
        - observation: 下一状态
        - reward: 奖励值
        - done: 是否结束
        - info: 额外信息
    """
    if self.is_done:
      raise RuntimeError("环境已结束，请先调用 reset()")

    # 记录当前资产（用于计算奖励）
    prev_datetime = datetime.combine(
      self.topn_list[self.current_step].base_date,
      datetime.min.time()
    )
    prev_assets = self.account.calc_assets(prev_datetime)

    # 1. 执行卖出动作
    stocks_to_sell = action.get('stocks_to_sell', [])
    for stock_code, shares in stocks_to_sell:
      if shares <= 0:
        continue
      pos = self.account.get_position(stock_code)
      if not pos or pos['volume'] < shares:
        testback_logger.warning(
          f"卖出 {stock_code} 失败：持仓不足 "
          f"(需要{shares}, 实际{pos['volume'] if pos else 0})"
        )
        continue

      sell_price = self.stock_prices.get(stock_code)
      if not sell_price:
        testback_logger.warning(f"卖出 {stock_code} 失败：无价格数据")
        continue

      try:
        self.account.sell_stock(
          code=stock_code,
          volume=shares,
          price=sell_price,
          sell_date=self.topn_list[self.current_step].base_date,
          clear_reason='RL策略'
        )
      except Exception as e:
        testback_logger.warning(f"卖出 {stock_code} 失败: {e}")

    # 2. 执行买入动作
    stocks_to_buy = action.get('stocks_to_buy', [])
    for stock_code, shares in stocks_to_buy:
      if shares <= 0:
        continue

      buy_price = self.stock_prices.get(stock_code)
      if not buy_price:
        testback_logger.warning(f"买入 {stock_code} 失败：无价格数据")
        continue

      try:
        self.account.buy_stock(
          code=stock_code,
          volume=shares,
          price=buy_price,
          buy_date=self.topn_list[self.current_step].base_date
        )
      except Exception as e:
        testback_logger.debug(f"买入 {stock_code} 失败: {e}")

    # 3. 计算当前资产和奖励
    current_datetime = datetime.combine(
      self.topn_list[self.current_step].base_date,
      datetime.min.time()
    )
    current_assets = self.account.calc_assets(current_datetime)

    # 计算策略收益率
    portfolio_return = (
        (current_assets['total_asset'] - prev_assets['total_asset'])
        / prev_assets['total_asset'] * 100
    )

    # 计算奖励
    if self.use_alpha_reward:
      # 使用超额收益（alpha）作为奖励
      baseline_return = self.baseline_returns[self.current_step] if self.current_step < len(self.baseline_returns) else 0.0
      reward = portfolio_return - baseline_return  # alpha = 策略收益 - 基准收益

      testback_logger.debug(
        f"[{self.topn_list[self.current_step].base_date}] "
        f"策略收益: {portfolio_return:.2f}%, "
        f"基准收益: {baseline_return:.2f}%, "
        f"Alpha: {reward:.2f}%"
      )
    else:
      # 使用绝对收益率作为奖励
      reward = portfolio_return

    # 记录
    self.episode_rewards.append(reward)
    self.episode_actions.append(action)

    # 4. 移动到下一步
    self.current_step += 1

    # 5. 检查是否结束
    if self.current_step >= self.max_steps:
      self.is_done = True

    # 6. 获取下一状态观测
    next_observation = self._get_observation() if not self.is_done else {}

    # 7. 构建info字典
    info = {
      'date': self.topn_list[self.current_step - 1].base_date,
      'cash': current_assets['cash'],
      'market_value': current_assets['market_value'],
      'total_asset': current_assets['total_asset'],
      'positions_count': len(self.account.positions),
      'step_reward': reward,
      'cumulative_return': (
          (current_assets['total_asset'] - self.account.init_cash)
          / self.account.init_cash * 100
      )
    }

    return next_observation, reward, self.is_done, info

  def _get_observation(self) -> Dict[str, Any]:
    """
    获取当前状态观测

    Returns:
        观测字典，包含：
        - candidate_stocks: 候选股票列表
        - factor_scores: 因子得分矩阵 {stock: {factor: score}}
        - current_positions: 当前持仓 {stock: shares}
        - cash: 可用资金
        - prices: 股票价格 {stock: price}
        - date: 当前日期
    """
    if self.current_step >= self.max_steps:
      return {}

    topn = self.topn_list[self.current_step]
    trade_date = topn.base_date

    # 1. 获取候选股票和因子得分
    try:
      # 使用 get_ordered_stocks 获取排序后的股票
      self.candidate_stocks = topn.get_ordered_stocks(
        n=self.rank_n,
        weights=self.weights or {k: 1.0 for k in topn.factor_scores.keys()},
        temperatures=self.temperatures or {k: 1.0 for k in topn.factor_scores.keys()},
        norm_method=self.norm_method
      )
    except Exception as e:
      testback_logger.warning(f"获取候选股票失败: {e}")
      self.candidate_stocks = []

    # 2. 获取股票价格
    self.stock_prices = {}
    for stock in self.candidate_stocks:
      try:
        trade_datetime = datetime.combine(trade_date, datetime.min.time())
        data = get_market_data_from_cache(stock, 1, trade_datetime)
        if data is not None and len(data) > 0:
          self.stock_prices[stock] = float(data.iloc[-1]['close'])
      except Exception as e:
        testback_logger.warning(f"{stock} 价格获取失败: {e}")

    # 3. 提取因子得分（归一化后的）
    factor_scores = {}
    for stock in self.candidate_stocks:
      factor_scores[stock] = {}
      for factor_name, scores in topn.factor_scores.items():
        if stock in scores:
          result = scores[stock]
          
          # 处理不同类型的结果
          if isinstance(result, dict):
            # 如果是字典，尝试获取 norm_score 或 score
            factor_scores[stock][factor_name] = result.get('norm_score', result.get('score', 0.0))
          elif hasattr(result, 'norm_score') and result.norm_score is not None:
            # FactorResult对象，优先使用归一化分数
            factor_scores[stock][factor_name] = result.norm_score
          elif hasattr(result, 'score'):
            # FactorResult对象，使用原始分数
            factor_scores[stock][factor_name] = result.score
          else:
            # 直接是数值
            factor_scores[stock][factor_name] = float(result)
    
    # 4. 获取当前持仓
    current_positions = {
      stock: pos['volume']
      for stock, pos in self.account.positions.items()
    }

    # 5. 构建观测字典
    observation = {
      'candidate_stocks': self.candidate_stocks,
      'factor_scores': factor_scores,
      'current_positions': current_positions,
      'cash': self.account.current_cash,
      'prices': self.stock_prices,
      'date': trade_date,
      'step': self.current_step,
    }

    return observation

  def get_observation_array(self) -> np.ndarray:
    """
    将观测转换为numpy数组（用于神经网络输入）

    Returns:
        观测向量
    """
    obs = self._get_observation()

    # 特征向量设计（示例）：
    # [现金占比, 持仓占比, 因子得分平均值, ...]
    total_asset = self.account.calc_assets(
      datetime.combine(
        self.topn_list[self.current_step].base_date,
        datetime.min.time()
      )
    )['total_asset']

    features = [
      obs['cash'] / total_asset,  # 现金占比
      len(obs['current_positions']) / self.rank_n,  # 持仓比例
    ]

    # 添加因子得分的统计特征
    if obs['factor_scores']:
      all_scores = []
      for stock_scores in obs['factor_scores'].values():
        all_scores.extend(stock_scores.values())
      if all_scores:
        features.extend([
          np.mean(all_scores),
          np.std(all_scores),
          np.max(all_scores),
          np.min(all_scores),
        ])
      else:
        features.extend([0.0, 0.0, 0.0, 0.0])
    else:
      features.extend([0.0, 0.0, 0.0, 0.0])

    return np.array(features, dtype=np.float32)

  def done(self) -> bool:
    """检查环境是否结束"""
    return self.is_done

  def get_episode_summary(self) -> Dict[str, Any]:
    """
    获取完整回合的统计信息

    Returns:
        回合统计信息
    """
    if not self.is_done:
      testback_logger.warning("回合尚未结束")

    final_datetime = datetime.combine(
      self.topn_list[-1].base_date,
      datetime.min.time()
    )
    final_assets = self.account.calc_assets(final_datetime)

    total_return = (
        (final_assets['total_asset'] - self.account.init_cash)
        / self.account.init_cash * 100
    )

    # 计算夏普比率（假设无风险利率为0）
    if len(self.episode_rewards) > 1:
      sharpe_ratio = (
          np.mean(self.episode_rewards) / (np.std(self.episode_rewards) + 1e-8)
      )
    else:
      sharpe_ratio = 0.0

    # 计算基准总收益率和超额收益
    baseline_total_return = 0.0
    alpha_total = 0.0

    if self.use_alpha_reward and self.baseline_returns:
      # 累计基准收益率（复利）
      baseline_cumulative = 1.0
      for baseline_return in self.baseline_returns[:self.current_step]:
        baseline_cumulative *= (1 + baseline_return / 100)
      baseline_total_return = (baseline_cumulative - 1.0) * 100

      # 超额收益 = 策略收益 - 基准收益
      alpha_total = total_return - baseline_total_return

    summary = {
      'total_steps': self.current_step,
      'init_cash': self.account.init_cash,
      'final_cash': final_assets['cash'],
      'final_market_value': final_assets['market_value'],
      'final_total_asset': final_assets['total_asset'],
      'total_return': total_return,
      'sharpe_ratio': sharpe_ratio,
      'total_reward': sum(self.episode_rewards),
      'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
      'max_reward': max(self.episode_rewards) if self.episode_rewards else 0.0,
      'min_reward': min(self.episode_rewards) if self.episode_rewards else 0.0,
      'cleared_positions_count': len(self.account.cleared_positions),
      'current_positions_count': len(self.account.positions),
    }

    # 添加alpha相关指标
    if self.use_alpha_reward:
      summary.update({
        'baseline_return': baseline_total_return,
        'alpha': alpha_total,
        'info_ratio': alpha_total / (np.std(self.episode_rewards) + 1e-8) if self.episode_rewards else 0.0,
      })

    return summary

def create_rl_env_from_cache(
    weights: Optional[Dict[str, float]] = None,
    rank_n: int = 20,
    **env_kwargs
) -> RLEnv:
  """
  从共享内存缓存创建RL环境（用于多进程场景）

  Args:
      weights: 因子权重
      rank_n: 选股数量
      **env_kwargs: 其他环境参数

  Returns:
      RLEnv实例
  """
  from utils.shared_memory import SharedMemoryCache

  # 从共享内存读取TopN数据
  testback_cache = SharedMemoryCache('testback_cache', compress_level=6)
  topn_list = testback_cache.get('topn_data')

  if not topn_list:
    raise RuntimeError("共享内存中没有TopN数据，请先调用 testback_cache.put('topn_data', data)")

  # 创建环境
  env = RLEnv(
    topn_list=topn_list,
    rank_n=rank_n,
    weights=weights,
    **env_kwargs
  )

  return env
