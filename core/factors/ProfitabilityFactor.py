# coding:utf-8
"""
盈利能力因子（方案B - 因子1）

输入指标：
- ROE（净资产收益率）
- 净利率
- 毛利率

输出：盈利能力分数（连续分数，正向因子，分数越高越好）
使用分位数截断+Sigmoid连续映射，确保每个指标贡献范围一致
"""
from .helpers import *
import numpy as np
from core.database.financial import (
    get_financial_indicator,
    get_roe,
    get_financial_indicator as get_financial
)


def continuous_score(value, p1, p99, center, scale=10, reverse=False):
    """
    连续评分函数：分位数截断 + Sigmoid映射
    
    Args:
        value: 指标值
        p1: 1%分位数（下限，截断极端值）
        p99: 99%分位数（上限，截断极端值）
        center: 中心点（优秀标准）
        scale: 缩放系数，控制曲线陡峭程度
        reverse: 是否反向（值越小分数越高）
    
    Returns:
        0-1之间的连续分数
    """
    # 1. 截断极端值（使用1%和99%分位数）
    value_clipped = np.clip(value, p1, p99)
    
    # 2. 归一化到中心点附近
    if reverse:
        x = (center - value_clipped) / center * scale
    else:
        x = (value_clipped - center) / center * scale
    
    # 3. Sigmoid映射到0-1范围
    x_clipped = np.clip(x, -20, 20)  # 防止exp溢出
    score = 1.0 / (1.0 + np.exp(-x_clipped))
    
    return score


class ProfitabilityFactor(BaseFactor):
    """
    盈利能力因子

    使用连续Sigmoid函数评分（基于实际数据的分位数）：
    - ROE：1%分位=-46.23, 99%分位=28.88, 中心点=15%, 权重40%
    - 净利率：1%分位=-107.22, 99%分位=54.90, 中心点=10%, 权重30%
    - 毛利率：1%分位=-6.61, 99%分位=87.50, 中心点=30%, 权重30%

    每个指标先映射到0-1范围，然后加权平均，确保贡献一致
    """

    def __init__(self):
        super().__init__()

    def calc(self, ctx: FactorCtx) -> FactorResult:
        query_date = ctx.base_time.date()
        stock_code = ctx.code

        # 获取财务指标
        roe = get_roe(stock_code, query_date)
        net_profit = get_financial(stock_code, query_date, 'net_profit', 'PershareIndex')
        gross_profit = get_financial(stock_code, query_date, 'gross_profit', 'PershareIndex')

        if roe is None:
            raise ValueError(f"无ROE数据: {stock_code}")
        if net_profit is None:
            raise ValueError(f"无净利率数据: {stock_code}")
        if gross_profit is None:
            raise ValueError(f"无毛利率数据: {stock_code}")

        # 使用连续函数计算每个指标的分数（都映射到0-1范围）
        roe_score = continuous_score(roe, p1=-46.23, p99=28.88, center=15, scale=10)
        net_profit_score = continuous_score(net_profit, p1=-107.22, p99=54.90, center=10, scale=10)
        gross_profit_score = continuous_score(gross_profit, p1=-6.61, p99=87.50, center=30, scale=10)

        # 加权平均（每个指标贡献范围都是0-1，不会因为范围差距而丢失信息）
        score = roe_score * 0.4 + net_profit_score * 0.3 + gross_profit_score * 0.3

        return FactorResult(score=score, err=None)

