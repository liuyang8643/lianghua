import numpy as np
from core.factors.helpers import BaseFactor, FactorResult, FactorCtx

class SmallCap(BaseFactor):
  """小盘股因子 - 基于成交额近似市值"""

  def calc(self, ctx: FactorCtx) -> FactorResult:
    # 使用20天数据，平衡稳定性和覆盖度
    try:
      history_data = ctx.get_daily_data(60)
    except ValueError as e:
      return FactorResult(score=None, err=str(e))

    # 检查数据是否存在
    if history_data is None or history_data.empty:
      return FactorResult(score=None, err="no data")

    amount_col = history_data.get('amount')
    if amount_col is None or amount_col.empty:
      return FactorResult(score=None, err="no amount data")

    avg_amount_yi = amount_col.values.mean() / 1e8

    # 成交额越小分数越高 (指数衰减)
    score = 100 * np.exp(-(avg_amount_yi / 5))

    return FactorResult(
      score=score,
      raw_value=avg_amount_yi,
      metadata={'avg_amount_yi': avg_amount_yi}
    )
