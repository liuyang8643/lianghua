from .helpers import *

class CCI(BaseFactor):
  """
  CCI顺势指标因子
  CCI = (TP - MA) / (0.015 * MD)
  """

  def __init__(self):
    super().__init__()

  def calc(self, ctx: FactorCtx) -> FactorResult:
    cci_value = ctx.get_cci()
    return FactorResult(score=cci_value, err=None)
