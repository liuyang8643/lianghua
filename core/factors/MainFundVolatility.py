from .helpers import *
from core.database.money_flow import *
import numpy as np
from datetime import datetime

class MainFundVolatility(BaseFactor):
  """
  主力资金净流入波动率因子

  计算公式：
  波动率 = std(过去20天主力资金净流入/当日成交额)
  """

  def __init__(self):
    super().__init__()
    self.period = 20

  def calc(self, ctx: FactorCtx) -> FactorResult:
    try:
      # 1. 获取过去 period 天的交易日期
      hist_data = ctx.get_daily_data(self.period)

      if hist_data is None or len(hist_data) < self.period:
        return FactorResult(score=None, err=ValueError(f"历史数据 {len(hist_data) if hist_data is not None else 0} 不足 {self.period}天: {ctx.code}"))

      # 2. 获取每日有效分数
      valid_scores = []
      for i in range(len(hist_data)):
        # 将字符串日期转换为 datetime 对象
        trade_date = datetime.strptime(hist_data.index[i], '%Y%m%d')
        net_inflow = get_main_fund_net_inflow(ctx.code, trade_date)
        score = net_inflow / hist_data.iloc[i]['amount'] if net_inflow is not None else None

        if score is not None:
          valid_scores.append(score)
        else:
          return FactorResult(score=None, err=ValueError(f"主力资金数据 {len(valid_scores)} 不足 {self.period}天: {ctx.code}"))

      # 3. 计算标准差（取负值，波动率越大分数越低）
      volatility = -np.std(valid_scores, ddof=1)

      return FactorResult(score=volatility, err=None)

    except Exception as e:
      return FactorResult(score=None, err=e)
