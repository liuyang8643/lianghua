from typing import Optional, TypedDict

from .indicators import FactorCtx

class FactorResult(TypedDict):
  # 计算得分，如果计算出错则为 None
  score: Optional[float]

class BaseFactor:
  def calc(self, ctx: FactorCtx) -> FactorResult:
    """
    计算因子得分
    :param ctx: 上下文
    :return: 判断结果
    """
    raise NotImplementedError("必须实现判断逻辑")

