from .helpers import *

class BBI(BaseFactor):
  """
  BBI多空指数因子
  BBI = (MA3 + MA6 + MA12 + MA24) / 4
  """

  def __init__(self):
    super().__init__()

  def calc(self, ctx: FactorCtx) -> FactorResult:
    try:
      bbi_value = ctx.get_bbi()  # 直接获取当前BBI值
      current_price = ctx.get_current_price()
      # 计算价格与BBI的偏离度作为得分
      score = (current_price - bbi_value) / bbi_value
      return FactorResult(score=score, err=None)
    except Exception as e:
      return FactorResult(score=None, err=e)
