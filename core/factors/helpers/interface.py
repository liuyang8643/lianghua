from enum import Enum
from typing import Optional, TypedDict

from .ctx import JudgeCtx

class TradeSignal(Enum):
  # 买入
  Buy = 1
  # 卖出
  Sell = -1
  # 观望
  Hold = 0

class JudgeResult(TypedDict):
  # 判断倾向，1为买入，-1为卖出，0为观望
  signal: Optional[TradeSignal]
  # 预期价格
  trigger_price: Optional[float]
  # 因子名称
  factor: str
  # 具体的判断依据
  reason: str

class BaseFactor:
  def real_trade(self, ctx: JudgeCtx) -> JudgeResult:
    """
    实盘判断
    :param ctx: 上下文
    :return: 判断结果
    """
    raise NotImplementedError("必须实现实盘判断逻辑")

  def mock_trade_daily(self, ctx: JudgeCtx) -> JudgeResult:
    """
    模拟日线交易
    :param ctx: 上下文
    :return: 模拟交易结果
    """
    raise NotImplementedError("必须实现日线模拟交易逻辑")
