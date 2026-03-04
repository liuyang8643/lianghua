from .helpers import *
from core.database.money_flow import *
import numpy as np
from datetime import datetime

class MainFundFlow(BaseFactor):
  """
  主力资金流向因子

  计算公式：
  ratio_i = net_inflow_i / amount_i
  ma5 = mean(ratio[-5:])
  ma20 = mean(ratio[-20:])
  momentum = ma5 - ma20
  score = (ma5 * 0.6 + momentum * 0.4) * 1000
  """

  def __init__(self):
    super().__init__()
    self.period = 20

  def calc(self, ctx: FactorCtx) -> FactorResult:
    try:
      hist_data = ctx.get_daily_data(self.period)

      if hist_data is None or len(hist_data) < self.period:
        return FactorResult(score=None, err=ValueError(f"历史数据 {len(hist_data) if hist_data is not None else 0} 不足 {self.period}天: {ctx.code}"))

      ratios = []
      for i in range(len(hist_data)):
        trade_date = datetime.strptime(hist_data.index[i], '%Y%m%d')
        net_inflow = get_main_fund_net_inflow(ctx.code, trade_date)

        if net_inflow is None:
          return FactorResult(score=None, err=ValueError(f"主力资金数据缺失 第{i}天: {ctx.code}"))

        amount = hist_data.iloc[i]['amount']
        if amount is None or amount <= 0:
          return FactorResult(score=None, err=ValueError(f"成交额数据异常 第{i}天: {ctx.code}"))

        ratios.append(net_inflow / amount)

      ratios = np.array(ratios)
      ma5 = np.mean(ratios[-5:])
      ma20 = np.mean(ratios)
      momentum = ma5 - ma20
      score = (ma5 * 0.6 + momentum * 0.4) * 1000

      return FactorResult(score=score, err=None)

    except Exception as e:
      return FactorResult(score=None, err=e)
