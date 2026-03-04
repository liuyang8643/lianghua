import numpy as np
from .helpers import BaseFactor, FactorResult, FactorCtx
from .helpers.cache import cached_factor
from core.database.financial import get_financial_indicator


class SmallCapV2(BaseFactor):
  """小盘股因子V2 - 基于总股本×股价计算真实总市值，市值越小分数越高"""

  @cached_factor('SmallCapV2')
  def calc(self, ctx: FactorCtx) -> FactorResult:
    query_date = ctx.base_time.date()

    cap_stk = get_financial_indicator(ctx.code, query_date, 'cap_stk', 'Balance')

    if cap_stk is None or cap_stk <= 0:
      return FactorResult(score=None, err='无cap_stk数据')

    current_price = ctx.get_current_price()
    total_market_cap = cap_stk * current_price
    market_cap_yi = total_market_cap / 1e8

    score = float(np.clip(100 * np.exp(-(market_cap_yi / 30)), 0, 100))

    return FactorResult(
      score=score,
      err=None,
      metadata={
        'market_cap_yi': market_cap_yi,
        'cap_stk': cap_stk,
        'current_price': current_price,
      }
    )
