from .helpers import *
from core.database.money_flow import *


class RetailFlow(BaseFactor):
    """
    散户资金占比因子

    计算公式：
    散户占比 = (小单买入金额 + 小单卖出金额) / 当日总成交额

    说明：
    - 散户定义为小单交易（通常为散户行为）
    - 占比越高表示散户参与度越高
    - 需要配合 BatchNorm 进行跨股票归一化使用
    """

    def __init__(self):
        super().__init__()

    def calc(self, ctx: FactorCtx) -> FactorResult:
        # 1. 获取散户资金金额（小单买入 + 小单卖出）
        retail_amount = get_retail_flow_amount(ctx.code, ctx.base_time)
        if retail_amount is None:
            raise ValueError(f"无资金流向数据: {ctx.code}")

        # 2. 获取当日总成交额（从 K 线数据）
        today_data = ctx.get_today_data()
        total_amount = float(today_data['amount'])

        if total_amount <= 0:
            raise ValueError(f"成交额异常: {ctx.code}")

        # 3. 计算散户占比（百分比形式）
        retail_ratio = (retail_amount / total_amount) * 100

        return FactorResult(score=retail_ratio, err=None)
