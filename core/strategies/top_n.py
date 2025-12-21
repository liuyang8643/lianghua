from datetime import datetime
from typing import Optional, Dict, List

from core.factors import MACD, BBI, CCI, Fundamental, TRIXFactor, MOMFactor, ADXFactor, FactorCtx, FactorResult, BatchNormFactor
from utils.parallel import batch_run_threads
from utils.stock.format import format_qmt_date
from ..logger import core_logger

class TopN:
  """多因子选股策略 - 选取综合得分排名前N的股票"""

  def __init__(
      self,
      stock_list: List[str],
      base_date: datetime,  # 只接受 datetime
  ):
    """
    初始化TopN策略

    Args:
        stock_list: 股票池列表
        base_date: 基准日期（必须是datetime对象）
    """
    self.stock_list = stock_list
    self.base_date = base_date

    # 初始化因子列表（暂时移除Fundamental以加快测试）
    self.factors = [
      MACD(),
      BBI(),
      CCI(),
      # Fundamental(),  # 暂时注释，API调用慢
      TRIXFactor(),
      MOMFactor(),
      ADXFactor(),
    ]

    # ✅ 修复：正确的数据结构
    # {factor_name: {stock_code: FactorResult}}
    self.factor_scores: Dict[str, Dict[str, FactorResult]] = {
      factor.__class__.__name__: {}
      for factor in self.factors
    }

    # 多线程计算因子原始分数
    batch_run_threads(
      func=self._calculate_factor_score,
      args_list=[[stock_code] for stock_code in self.stock_list],
      max_workers=64,
    )

  def _calculate_factor_score(self, stock_code: str):
    """
    计算单个股票的所有因子分数（用于并行执行）

    Args:
        stock_code: 股票代码
    """
    ctx = FactorCtx(stock_code, self.base_date)

    # 计算所有因子
    for f in self.factors:
      factor_name = f.__class__.__name__
      result = f.calc(ctx)
      self.factor_scores[factor_name][stock_code] = result

  def _count_valid_stocks(self) -> int:
    """统计至少有一个有效因子分数的股票数量"""
    valid_stocks = set()

    for factor_name, scores in self.factor_scores.items():
      for stock_code, result in scores.items():
        if result['err'] is None and result['score'] is not None:
          valid_stocks.add(stock_code)

    return len(valid_stocks)

  def get_normalized_scores(
      self,
      temperatures: Dict[str, float],
      method: str = 'softmax'
  ) -> Dict[str, Dict[str, float]]:
    """
    对所有因子进行批量归一化

    Args:
        temperatures: 因子温度参数字典 {factor_name: temperature}
        method: 归一化方法 ('softmax', 'minmax', 'zscore', 'rank')

    Returns:
        归一化后的分数 {factor_name: {stock_code: norm_score}}
    """
    normalized = {}

    for factor_name, raw_scores in self.factor_scores.items():
      # 获取该因子的温度参数（默认1.0）
      temperature = temperatures.get(factor_name, 1.0)

      # 批量归一化
      norm_scores = BatchNormFactor.batch_normalize(
        raw_scores=raw_scores,
        temperature=temperature,
        method=method
      )

      normalized[factor_name] = norm_scores

      core_logger.debug(
        f"因子 {factor_name}: 温度={temperature}, "
        f"有效股票数={len(norm_scores)}"
      )

    return normalized

  def get_ordered_stocks(
      self,
      n: int,
      weights: Dict[str, float],
      temperatures: Dict[str, float],
      norm_method: str = 'softmax'
  ) -> List[str]:
    """
    返回按综合得分排序的前N只股票

    Args:
        n: 选取数量
        weights: 因子权重字典 {factor_name: weight}
        temperatures: 因子温度字典 {factor_name: temperature}
        norm_method: 归一化方法

    Returns:
        排序后的股票代码列表（长度<=n）
    """
    # 1. 批量归一化
    normalized_scores = self.get_normalized_scores(temperatures, norm_method)

    # 2. 加权求和
    final_scores: Dict[str, float] = {}

    for stock_code in self.stock_list:
      score = 0.0
      valid_factors = 0

      for factor_name in self.factor_scores.keys():
        # 检查该股票是否有该因子的归一化分数
        if stock_code in normalized_scores[factor_name]:
          weight = weights.get(factor_name, 0.0)
          norm_score = normalized_scores[factor_name][stock_code]
          score += norm_score * weight
          valid_factors += 1

      # 只有至少有一个有效因子分数的股票才参与排序
      if valid_factors > 0:
        final_scores[stock_code] = score

    if len(final_scores) == 0:
      core_logger.warning(
        f"没有任何股票有有效的因子分数！日期: {format_qmt_date(self.base_date)}"
      )
      return []

    # 3. 排序并选取 Top N
    sorted_stocks = sorted(
      final_scores.items(),
      key=lambda x: x[1],
      reverse=True
    )

    top_n = [stock for stock, score in sorted_stocks[:n]]

    core_logger.info(
      f"{format_qmt_date(self.base_date)} TopN选股完成: "
      f"候选{len(final_scores)}只, 选出{len(top_n)}只"
    )

    if len(top_n) > 0:
      # 输出前5名的分数（用于调试）
      top_5 = sorted_stocks[:min(5, len(sorted_stocks))]
      core_logger.debug(
        f"Top 5: " +
        ", ".join([f"{stock}({score:.6f})" for stock, score in top_5])
      )

    return top_n

  def get_factor_scores_summary(self) -> Dict[str, dict]:
    """
    获取因子分数摘要（用于调试和分析）

    Returns:
        {
            factor_name: {
                'mean': 平均分,
                'std': 标准差,
                'min': 最小分,
                'max': 最大分,
                'valid_count': 有效数量,
            }
        }
    """
    import numpy as np

    summary = {}

    for factor_name, scores in self.factor_scores.items():
      valid_scores = [
        result['score']
        for result in scores.values()
        if result['err'] is None and result['score'] is not None
      ]

      if len(valid_scores) > 0:
        summary[factor_name] = {
          'mean': float(np.mean(valid_scores)),
          'std': float(np.std(valid_scores)),
          'min': float(np.min(valid_scores)),
          'max': float(np.max(valid_scores)),
          'valid_count': len(valid_scores),
        }
      else:
        summary[factor_name] = {
          'mean': 0.0,
          'std': 0.0,
          'min': 0.0,
          'max': 0.0,
          'valid_count': 0,
        }

    return summary
