from typing import Optional, TypedDict
from core.factors.helpers.indicators import FactorCtx

class FactorResult(TypedDict):
  # 计算得分，如果计算出错则为 None
  score: Optional[float]
  # 可选的错误信息
  err: Optional[Exception]

class BaseFactor:
  # 因子需要的历史数据天数（向前看），用于自动计算预热期
  # 子类应覆盖此值，如：hist_days = 20
  hist_days: int = 0

  def calc(self, ctx: FactorCtx) -> FactorResult:
    """
    计算因子得分
    :param ctx: 上下文
    :return: 判断结果
    """
    raise NotImplementedError("必须实现判断逻辑")
