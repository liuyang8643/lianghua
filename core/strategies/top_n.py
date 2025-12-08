from datetime import datetime
from typing import Optional

from core.factors import MACD, BBI, CCI, FactorCtx, FactorResult
from utils.parallel import batch_run_threads
from utils.stock.format import format_qmt_date
from ..logger import core_logger

class TopN:
  """ 给定股票池中，选取市值排名前N的股票 """

  def __init__(
      self, stock_list: list[str],
      base_date: datetime,
  ):
    self.stock_list = stock_list
    self.base_date = base_date
    self.factors = [
      # TODO 添加更多因子
      MACD(),
      BBI(),
      CCI(),
    ]
    self.factor_scores: Optional[dict[str, FactorResult]] = {}  # 因子名 -> 得分

    core_logger.debug(f"TopN 计算 {format_qmt_date(self.base_date)} 因子分数，共计{len(self.stock_list)}只股票")
    # 多线程计算因子原始分数
    batch_run_threads(
      func=self._calculate_factor_score,
      args_list=[[stock_code] for stock_code in self.stock_list],
      max_workers=64,  # 最大并发线程数
    )

    core_logger.debug(f"TopN 计算 {format_qmt_date(self.base_date)} 因子分数完成")

  def _calculate_factor_score(self, stock_code: str):
    """ 计算单个股票的综合得分（用于并行执行）
    
    :param stock_code: 股票代码
    :return: (股票代码, 综合得分)
    """
    ctx = FactorCtx(stock_code, self.base_date)

    # 因子计算优化：如果某个因子失败，继续计算其他因子
    for f in self.factors:
      factor_name = f.__class__.__name__
      try:
        self.factor_scores[factor_name] = f.calc(ctx)
      except Exception as e:
        # 忽略单个因子计算失败
        core_logger.warning(f"股票{stock_code}因子{factor_name}计算错误！忽略该因子分数: {e}")
        pass
