import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import pearsonr

from core import FactorCtx, allow_buy_stock_code_list, core_logger, get_full_market_data, get_stock_detail
from utils.stock.format import get_stock_desc
from utils.stock.time import get_trading_date_span

@dataclass
class StockCorrelation:
  """单只股票的相关性结果"""
  stock_code: str
  stock_name: str
  correlation: Optional[float]  # 皮尔逊相关系数
  p_value: Optional[float]  # 显著性检验p值
  sample_count: int  # 有效样本数量
  error: Optional[str] = None  # 错误信息

@dataclass
class FactorCorrelationReport:
  """因子相关性报告"""
  factor_name: str
  start_date: date
  end_date: date
  total_stocks: int
  valid_stocks: int  # 有效计算的股票数
  stock_correlations: list[StockCorrelation]
  avg_correlation: float  # 平均相关系数（仅统计有效值）
  median_correlation: float  # 中位数相关系数

def _calculate_single_stock_correlation(
    stock_code: str,
    factor_cls,
    date_list: list[date]
) -> StockCorrelation:
  """
  计算单只股票的因子与收盘价相关性（用于并行执行）

  Args:
      stock_code: 股票代码
      factor_cls: 因子类
      date_list: 交易日期列表

  Returns:
      StockCorrelation: 相关性结果
  """
  try:
    # 禁用xtquant的hello打印
    from xtquant import xtdata
    xtdata.enable_hello = False

    # 获取股票详情
    detail = get_stock_detail(stock_code)
    if detail is None:
      return StockCorrelation(
        stock_code=stock_code,
        stock_name="未知",
        correlation=None,
        p_value=None,
        sample_count=0,
        error="无法获取股票详情"
      )

    stock_name = detail.get('InstrumentName', stock_code)

    # 获取完整历史数据
    full_data = get_full_market_data(stock_code, '1d')
    if full_data is None or full_data.empty:
      return StockCorrelation(
        stock_code=stock_code,
        stock_name=stock_name,
        correlation=None,
        p_value=None,
        sample_count=0,
        error="无法获取历史数据"
      )

    # 创建时间戳到收盘价的映射
    time_to_close = {}
    for _, row in full_data.iterrows():
      dt = datetime.fromtimestamp(row['time'] / 1000)
      time_to_close[dt.date()] = row['close']

    # 计算每个日期的因子值和对应的收盘价
    factor_scores = []
    close_prices = []
    factor = factor_cls()

    for trading_date in date_list:
      # 检查是否有该日期的数据
      if trading_date not in time_to_close:
        continue

      try:
        # 创建因子上下文（使用当日收盘时间）
        ctx = FactorCtx(stock_code, datetime.combine(trading_date, datetime.max.time()))
        result = factor.calc(ctx)

        # 只有当因子计算成功且得分不为None时才记录
        if result['score'] is not None and not np.isnan(result['score']):
          factor_scores.append(result['score'])
          close_prices.append(time_to_close[trading_date])
      except Exception as e:
        # 单个日期计算失败，继续下一个
        continue

    # 检查是否有足够的样本
    if len(factor_scores) < 10:  # 至少需要10个样本才有统计意义
      return StockCorrelation(
        stock_code=stock_code,
        stock_name=stock_name,
        correlation=None,
        p_value=None,
        sample_count=len(factor_scores),
        error=f"有效样本不足（需要至少10个，实际{len(factor_scores)}个）"
      )

    # 计算皮尔逊相关系数
    correlation, p_value = pearsonr(factor_scores, close_prices)

    core_logger.info(f"股票 {get_stock_desc(detail)} 相关性计算完成: r={correlation:.4f}, p={p_value:.4f}, n={len(factor_scores)}")

    return StockCorrelation(
      stock_code=stock_code,
      stock_name=stock_name,
      correlation=correlation,
      p_value=p_value,
      sample_count=len(factor_scores)
    )

  except Exception as e:
    core_logger.error(f"计算股票 {stock_code} 相关性时出错: {e}")
    return StockCorrelation(
      stock_code=stock_code,
      stock_name="未知",
      correlation=None,
      p_value=None,
      sample_count=0,
      error=str(e)
    )

def calculate_factor_correlation(
    factor_cls,
    start_date: date,
    end_date: date,
    stock_codes: list[str] = None
) -> FactorCorrelationReport:
  """
  计算因子与股价的相关性

  Args:
      factor_cls: 因子类（如MACD）
      start_date: 开始日期
      end_date: 结束日期
      stock_codes: 股票代码列表，如果为None则使用允许买入的股票列表

  Returns:
      FactorCorrelationReport: 相关性报告
  """
  # 获取股票列表
  if stock_codes is None:
    stock_codes = allow_buy_stock_code_list()

  factor_name = factor_cls.__name__

  core_logger.info(f"开始计算因子 {factor_name} 的相关性分析...")
  core_logger.info(f"时间范围: {start_date} 至 {end_date}")
  core_logger.info(f"股票数量: {len(stock_codes)}")

  # 获取交易日期列表
  date_list = get_trading_date_span(start_date, end_date)
  core_logger.info(f"交易日数量: {len(date_list)}")

  # 并行计算每只股票的相关性
  worker_count = min(os.cpu_count() or 4, len(stock_codes))
  core_logger.info(f"使用 {worker_count} 个进程并行计算...")

  results = Parallel(n_jobs=worker_count, backend='loky')(
    delayed(_calculate_single_stock_correlation)(
      stock_code,
      factor_cls,
      date_list
    )
    for stock_code in stock_codes
  )

  # 统计有效结果
  valid_results = [r for r in results if r.correlation is not None]
  valid_correlations = [r.correlation for r in valid_results]

  # 计算平均相关性和中位数
  avg_correlation = np.mean(valid_correlations) if valid_correlations else 0.0
  median_correlation = np.median(valid_correlations) if valid_correlations else 0.0

  core_logger.info(f"计算完成！有效股票: {len(valid_results)}/{len(stock_codes)}")
  core_logger.info(f"平均相关系数: {avg_correlation:.4f}")
  core_logger.info(f"中位数相关系数: {median_correlation:.4f}")

  return FactorCorrelationReport(
    factor_name=factor_name,
    start_date=start_date,
    end_date=end_date,
    total_stocks=len(stock_codes),
    valid_stocks=len(valid_results),
    stock_correlations=results,
    avg_correlation=avg_correlation,
    median_correlation=median_correlation
  )
