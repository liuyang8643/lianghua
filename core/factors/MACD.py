from .helpers import *

class MACD(BaseFactor):
  """
  判断当前股票是否MAW金叉
  """

  def __init__(self):
    super().__init__()

  def calc(self, ctx: FactorCtx) -> FactorResult:
    macd, signal, hist = ctx.get_macd()
    return FactorResult(
      score=macd
    )
