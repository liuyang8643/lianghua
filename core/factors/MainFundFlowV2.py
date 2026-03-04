import numpy as np
from datetime import datetime
from .helpers import BaseFactor, FactorResult, FactorCtx
from core.database.money_flow import get_active_main_retail_net


def _calc_divergences(ctx: FactorCtx, period: int):
  """
  计算每日主散博弈差值序列
  divergence[i] = (主动大单净流入 - 散户净流入) / 当日成交额

  已经是无量纲比值，跨股票天然可比，无需额外归一化。
  Returns: np.ndarray 或 None
  """
  hist_data = ctx.get_daily_data(period)
  if hist_data is None or len(hist_data) < period:
    return None

  divergences = []
  for i in range(len(hist_data)):
    trade_date = datetime.strptime(hist_data.index[i], '%Y%m%d')
    result = get_active_main_retail_net(ctx.code, trade_date)
    if result is None:
      return None

    active_main_net, retail_net = result
    amount = hist_data.iloc[i]['amount']
    if amount is None or amount <= 0:
      return None

    divergences.append((active_main_net - retail_net) / amount)

  return np.array(divergences)


class MainFundFlowV2(BaseFactor):
  """
  主散博弈因子（特征3）

  score = mean(divergence[-5:])

  divergence[i] = (主动大单净流入 - 散户净流入) / amount
  - 主动大单：只取主动买/卖，排除被动单/对倒噪声
  - 散户净流入：小单买 - 小单卖（正=散户追涨，对主力是负信号）
  - 除以成交额后已是无量纲比值，跨股票可比
  """

  def __init__(self):
    super().__init__()
    self.period = 20
    self.ma_short = 5

  def calc(self, ctx: FactorCtx) -> FactorResult:
    try:
      divergences = _calc_divergences(ctx, self.period)
      if divergences is None:
        return FactorResult(score=None, err=ValueError(f'数据不足: {ctx.code}'))

      score = np.mean(divergences[-self.ma_short:])
      return FactorResult(score=float(score))

    except Exception as e:
      return FactorResult(score=None, err=e)


class MainFundFlowV3(BaseFactor):
  """
  主散博弈 + 资金动量因子（特征3 + 特征4）

  score = level + momentum
    level    = mean(divergence[-5:])
    momentum = mean(divergence[-5:]) - mean(divergence[-20:])

  两项单位相同（均为无量纲比值），等权相加无需额外缩放。
  """

  def __init__(self):
    super().__init__()
    self.period = 20
    self.ma_short = 5

  def calc(self, ctx: FactorCtx) -> FactorResult:
    try:
      divergences = _calc_divergences(ctx, self.period)
      if divergences is None:
        return FactorResult(score=None, err=ValueError(f'数据不足: {ctx.code}'))

      level    = np.mean(divergences[-self.ma_short:])
      momentum = np.mean(divergences[-self.ma_short:]) - np.mean(divergences)

      score = level + momentum
      return FactorResult(score=float(score))

    except Exception as e:
      return FactorResult(score=None, err=e)
