from core.factors import *

from datetime import datetime

from ._weights import FactorWights
from ..logger import core_logger

class TopN:
  """ 给定股票池中，选取市值排名前N的股票 """

  def __init__(self, stock_list: list[str], target_date: datetime):
    self.stock_list = stock_list
    self.target_date = target_date

  def get_factors(self) -> list[tuple[BaseFactor, float]]:
    return [
      # TODO 添加更多因子
      (MACD(), FactorWights['MACD']),
    ]

  def get_ordered_stocks(self) -> list[str]:
    """ 获取排序后的股票列表,从高到低返回N个股票代码 """
    stock_scores: list[tuple[str, float]] = []
    for stock_code in self.stock_list:
      ctx = FactorCtx(stock_code, self.target_date)
      total_score = 0.0
      for factor, weight in self.get_factors():
        try:
          result = factor.calc(ctx)
          score = result['score'] or 0.0
          # TODO 分数归一化处理
          total_score += score * weight
        except Exception as e:
          # 如果某个因子计算失败，则跳过该因子
          core_logger.warning(e)
          continue
      stock_scores.append((stock_code, total_score))

    return [
      stock_code for stock_code, score in sorted(
        stock_scores,
        key=lambda x: x[1],
        reverse=True
      )
    ]
