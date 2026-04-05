import numpy as np
from core.factors.helpers import BaseFactor, FactorResult, FactorCtx

MIN_RAW_PRICE = 2.0


class SmallCapKeepST(BaseFactor):
  
  """小盘股因子 - 保留普通ST，仅排除*ST/退市整理/终止上市"""

  hist_days = 60

  def calc(self, ctx: FactorCtx) -> FactorResult:
    from core.database.stock_name import is_star_st_at_date  # v3: date-aware cache
    if is_star_st_at_date(ctx.code, ctx.base_time):
      return FactorResult(score=None, err="star_st_stock")

    try:
      history_data = ctx.get_daily_data(60)
    except ValueError as e:
      return FactorResult(score=None, err=str(e))

    if history_data is None or history_data.empty:
      return FactorResult(score=None, err="no data")

    raw_close = ctx.get_raw_close()
    if raw_close is not None and raw_close < MIN_RAW_PRICE:
      return FactorResult(score=None, err=f"raw_price_{raw_close:.2f}<{MIN_RAW_PRICE}")

    amount_col = history_data.get('amount')
    if amount_col is None or amount_col.empty:
      return FactorResult(score=None, err="no amount data")

    avg_amount_yi = amount_col.values.mean() / 1e8

    score = 100 * np.exp(-(avg_amount_yi / 5))

    return FactorResult(
      score=score,
      err=None,
      raw_value=avg_amount_yi,
      metadata={'avg_amount_yi': avg_amount_yi}
    )
