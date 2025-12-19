import numpy as np
from core.factors.helpers import BaseFactor, FactorResult, FactorCtx, cached_factor

class SmallCap(BaseFactor):
  """小盘股因子 - 基于成交额近似市值"""

  @cached_factor('SmallCap')
  def calc(self, ctx: FactorCtx) -> FactorResult:
    history_data = ctx.get_daily_data(60)
    avg_amount_yi = history_data['amount'].values.mean() / 1e8

    # 过滤: 3000万 < 日均成交额 < 20亿
    if avg_amount_yi < 0.3 or avg_amount_yi > 20:
      return FactorResult(score=0, err=f"filtered:{avg_amount_yi:.2f}亿")

    # 成交额越小分数越高 (指数衰减)
    score = 100 * np.exp(-(avg_amount_yi / 5))

    return FactorResult(
      score=score,
      raw_value=avg_amount_yi,
      metadata={'avg_amount_yi': avg_amount_yi}
    )
