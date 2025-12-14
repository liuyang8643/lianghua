import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr

from core import FactorCtx, allow_buy_stock_code_list, core_logger, get_full_market_data, get_stock_detail
from utils.stock.time import get_trading_date_span

@dataclass
class StockFactorScore:
  """单只股票在某日的因子得分"""
  stock_code: str
  stock_name: str
  factor_score: Optional[float]  # 因子得分
  close_price: Optional[float]  # 当日收盘价
  future_close_price: Optional[float]  # T+M日收盘价
  return_rate: Optional[float]  # T+M日收益率
  error: Optional[str] = None

@dataclass
class DailyCorrelation:
  """单日的相关性结果"""
  trade_date: date
  m_days: int  # T+M日收益率的M值
  correlation: Optional[float]  # 皮尔逊相关系数（因子得分排名与收益率）
  rank_correlation: Optional[float]  # 斯皮尔曼等级相关系数
  p_value: Optional[float]  # 显著性检验p值
  valid_stock_count: int  # 有效股票数
  stock_scores: list[StockFactorScore]  # 当日所有股票的得分详情

@dataclass
class PeriodStatistics:
  """单个持有期的统计结果"""
  m_days: int  # T+M日收益率
  daily_correlations: list[DailyCorrelation]  # 每日相关性
  avg_correlation: float  # 平均相关系数
  median_correlation: float  # 中位数相关系数
  avg_rank_correlation: float  # 平均等级相关系数
  positive_days: int  # 正相关天数
  negative_days: int  # 负相关天数
  valid_days: int  # 有效交易日数

@dataclass
class FactorCorrelationReport:
  """因子相关性报告"""
  factor_name: str
  start_date: date
  end_date: date
  m_days_list: list[int]  # T+M日收益率列表
  total_stocks: int
  period_statistics: list[PeriodStatistics]  # 各持有期的统计结果

  # 兼容旧版本的属性
  @property
  def m_days(self) -> int:
    """返回第一个持有期（兼容性）"""
    return self.m_days_list[0] if self.m_days_list else 5

  @property
  def daily_correlations(self) -> list[DailyCorrelation]:
    """返回第一个持有期的日度相关性（兼容性）"""
    return self.period_statistics[0].daily_correlations if self.period_statistics else []

  @property
  def avg_correlation(self) -> float:
    """返回第一个持有期的平均相关系数（兼容性）"""
    return self.period_statistics[0].avg_correlation if self.period_statistics else 0.0

  @property
  def median_correlation(self) -> float:
    """返回第一个持有期的中位数相关系数（兼容性）"""
    return self.period_statistics[0].median_correlation if self.period_statistics else 0.0

  @property
  def avg_rank_correlation(self) -> float:
    """返回第一个持有期的平均等级相关系数（兼容性）"""
    return self.period_statistics[0].avg_rank_correlation if self.period_statistics else 0.0

  @property
  def positive_days(self) -> int:
    """返回第一个持有期的正相关天数（兼容性）"""
    return self.period_statistics[0].positive_days if self.period_statistics else 0

  @property
  def negative_days(self) -> int:
    """返回第一个持有期的负相关天数（兼容性）"""
    return self.period_statistics[0].negative_days if self.period_statistics else 0

  @property
  def valid_stocks(self) -> int:
    """平均有效股票数"""
    if not self.period_statistics or not self.period_statistics[0].daily_correlations:
      return 0
    return int(np.mean([dc.valid_stock_count for dc in self.period_statistics[0].daily_correlations]))

  @property
  def stock_correlations(self) -> list:
    """为了兼容旧版报告生成器，返回空列表"""
    return []

def _calculate_daily_correlation(
    trade_date: date,
    stock_codes: list[str],
    factor_cls,
    m_days: int,
    trading_dates: list[date]
) -> DailyCorrelation:
  """
  计算单日的因子得分排名与T+M日收益率的相关性（用于并行执行）

  Args:
      trade_date: 交易日期
      stock_codes: 股票代码列表
      factor_cls: 因子类
      m_days: T+M日收益率
      trading_dates: 所有交易日期列表（用于查找T+M日）

  Returns:
      DailyCorrelation: 当日相关性结果
  """
  try:
    # 禁用xtquant的hello打印
    from xtquant import xtdata
    xtdata.enable_hello = False

    # 找到T+M日
    try:
      current_idx = trading_dates.index(trade_date)
      future_idx = current_idx + m_days
      if future_idx >= len(trading_dates):
        return DailyCorrelation(
          trade_date=trade_date,
          m_days=m_days,
          correlation=None,
          rank_correlation=None,
          p_value=None,
          valid_stock_count=0,
          stock_scores=[]
        )
      future_date = trading_dates[future_idx]
    except (ValueError, IndexError):
      return DailyCorrelation(
        trade_date=trade_date,
        m_days=m_days,
        correlation=None,
        rank_correlation=None,
        p_value=None,
        valid_stock_count=0,
        stock_scores=[]
      )

    # 初始化因子
    factor = factor_cls()
    stock_scores = []

    # 计算每只股票的因子得分和收益率
    for stock_code in stock_codes:
      try:
        # 获取股票详情
        detail = get_stock_detail(stock_code)
        if detail is None:
          continue
        stock_name = detail.get('InstrumentName', stock_code)

        # 获取完整历史数据
        full_data = get_full_market_data(stock_code, '1d')
        if full_data is None or full_data.empty:
          continue

        # 创建时间戳到收盘价的映射
        time_to_close = {}
        for _, row in full_data.iterrows():
          dt = datetime.fromtimestamp(row['time'] / 1000)
          time_to_close[dt.date()] = row['close']

        # 检查是否有当日和T+M日的数据
        if trade_date not in time_to_close or future_date not in time_to_close:
          continue

        # 计算因子得分
        ctx = FactorCtx(stock_code, datetime.combine(trade_date, datetime.max.time()))
        result = factor.calc(ctx)

        if result['score'] is None or np.isnan(result['score']):
          continue

        # 计算收益率
        close_price = time_to_close[trade_date]
        future_close_price = time_to_close[future_date]
        return_rate = (future_close_price - close_price) / close_price

        stock_scores.append(StockFactorScore(
          stock_code=stock_code,
          stock_name=stock_name,
          factor_score=result['score'],
          close_price=close_price,
          future_close_price=future_close_price,
          return_rate=return_rate
        ))

      except Exception as e:
        # 单只股票计算失败，继续下一只
        continue

    # 检查是否有足够的有效样本
    if len(stock_scores) < 10:
      return DailyCorrelation(
        trade_date=trade_date,
        m_days=m_days,
        correlation=None,
        rank_correlation=None,
        p_value=None,
        valid_stock_count=len(stock_scores),
        stock_scores=stock_scores
      )

    # 提取因子得分和收益率
    factor_scores = np.array([s.factor_score for s in stock_scores])
    return_rates = np.array([s.return_rate for s in stock_scores])

    # 计算皮尔逊相关系数（因子得分与收益率）
    correlation, p_value = pearsonr(factor_scores, return_rates)

    # 计算斯皮尔曼等级相关系数
    rank_correlation, _ = spearmanr(factor_scores, return_rates)

    return DailyCorrelation(
      trade_date=trade_date,
      m_days=m_days,
      correlation=correlation,
      rank_correlation=rank_correlation,
      p_value=p_value,
      valid_stock_count=len(stock_scores),
      stock_scores=stock_scores
    )

  except Exception as e:
    core_logger.error(f"计算日期 {trade_date} 的相关性时出错: {e}")
    return DailyCorrelation(
      trade_date=trade_date,
      m_days=m_days,
      correlation=None,
      rank_correlation=None,
      p_value=None,
      valid_stock_count=0,
      stock_scores=[]
    )

def calculate_factor_correlation(
    factor_cls,
    start_date: date,
    end_date: date,
    m_days: int | list[int] = 5,
    stock_codes: list[str] = None
) -> FactorCorrelationReport:
  """
  计算因子得分排名与T+M日收益率的相关性

  Args:
      factor_cls: 因子类（如MACD）
      start_date: 开始日期
      end_date: 结束日期
      m_days: T+M日收益率，可以是单个整数或整数列表（如5或[1, 5, 10, 20]）
      stock_codes: 股票代码列表，如果为None则使用允许买入的股票列表

  Returns:
      FactorCorrelationReport: 相关性报告
  """
  # 获取股票列表
  if stock_codes is None:
    stock_codes = allow_buy_stock_code_list()

  # 将m_days转换为列表
  if isinstance(m_days, int):
    m_days_list = [m_days]
  else:
    m_days_list = m_days

  factor_name = factor_cls.__name__

  core_logger.info(f"开始计算因子 {factor_name} 的相关性分析...")
  core_logger.info(f"时间范围: {start_date} 至 {end_date}")
  core_logger.info(f"股票数量: {len(stock_codes)}")
  core_logger.info(f"T+M日收益率: {', '.join([f'T+{m}' for m in m_days_list])}日")

  # 获取交易日期列表
  date_list = get_trading_date_span(start_date, end_date)
  core_logger.info(f"交易日数量: {len(date_list)}")

  # 对每个持有期计算相关性
  period_statistics = []

  for m in m_days_list:
    core_logger.info(f"\n=== 开始计算 T+{m} 日收益率相关性 ===")

    # 并行计算每日的相关性
    worker_count = min(os.cpu_count() or 4, len(date_list))
    core_logger.info(f"使用 {worker_count} 个进程并行计算...")

    daily_results = Parallel(n_jobs=worker_count, backend='loky')(
      delayed(_calculate_daily_correlation)(
        trade_date,
        stock_codes,
        factor_cls,
        m,
        date_list
      )
      for trade_date in date_list
    )

    # 统计有效结果
    valid_results = [r for r in daily_results if r.correlation is not None]
    valid_correlations = [r.correlation for r in valid_results]
    valid_rank_correlations = [r.rank_correlation for r in valid_results]

    # 计算统计指标
    avg_correlation = np.mean(valid_correlations) if valid_correlations else 0.0
    median_correlation = np.median(valid_correlations) if valid_correlations else 0.0
    avg_rank_correlation = np.mean(valid_rank_correlations) if valid_rank_correlations else 0.0
    positive_days = sum(1 for c in valid_correlations if c > 0)
    negative_days = sum(1 for c in valid_correlations if c < 0)

    core_logger.info(f"T+{m}日 计算完成！有效交易日: {len(valid_results)}/{len(date_list)}")
    core_logger.info(f"T+{m}日 平均相关系数: {avg_correlation:.4f}")
    core_logger.info(f"T+{m}日 中位数相关系数: {median_correlation:.4f}")
    core_logger.info(f"T+{m}日 平均等级相关系数: {avg_rank_correlation:.4f}")
    core_logger.info(f"T+{m}日 正相关天数: {positive_days}, 负相关天数: {negative_days}")

    period_statistics.append(PeriodStatistics(
      m_days=m,
      daily_correlations=daily_results,
      avg_correlation=avg_correlation,
      median_correlation=median_correlation,
      avg_rank_correlation=avg_rank_correlation,
      positive_days=positive_days,
      negative_days=negative_days,
      valid_days=len(valid_results)
    ))

  return FactorCorrelationReport(
    factor_name=factor_name,
    start_date=start_date,
    end_date=end_date,
    m_days_list=m_days_list,
    total_stocks=len(stock_codes),
    period_statistics=period_statistics
  )
