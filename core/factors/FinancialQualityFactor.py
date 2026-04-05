# coding:utf-8
"""
财务质量因子（方案B - 因子3）

输入指标：
- 现金流（每股经营活动现金流量）
- 资产负债率

输出：财务质量分数（连续分数，正向因子，分数越高越好）
使用分位数截断+Sigmoid连续映射，确保每个指标贡献范围一致
"""
from .helpers import *
import numpy as np
from core.database.financial import (
    get_cash_flow_ps,
    get_gear_ratio
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


class FinancialQualityFactor(BaseFactor):
    """
    财务质量因子

    使用连续Sigmoid函数评分（基于实际数据的分位数）：
    - 现金流：1%分位=-2.75, 99%分位=4.97, 中心点=0.5, 权重50%（正向，越高越好）
    - 资产负债率：1%分位=4.93, 99%分位=93.80, 中心点=50%, 权重50%（反向，越低越好）

    每个指标先映射到0-1范围，然后加权平均，确保贡献一致
    """

    def __init__(self):
        super().__init__()

    def calc(self, ctx: FactorCtx) -> FactorResult:
        query_date = ctx.base_time.date()
        stock_code = ctx.code

        # 获取财务指标
        cash_flow_ps = get_cash_flow_ps(stock_code, query_date)
        gear_ratio = get_gear_ratio(stock_code, query_date)

        if cash_flow_ps is None:
            raise ValueError(f"无现金流数据: {stock_code}")
        if gear_ratio is None:
            raise ValueError(f"无资产负债率数据: {stock_code}")

        # 使用连续函数计算每个指标的分数（都映射到0-1范围）
        cash_flow_score = continuous_score(cash_flow_ps, p1=-2.75, p99=4.97, center=0.5, scale=5)
        gear_ratio_score = continuous_score(gear_ratio, p1=4.93, p99=93.80, center=50, scale=10, reverse=True)

        # 加权平均（每个指标贡献范围都是0-1，不会因为范围差距而丢失信息）
        score = cash_flow_score * 0.5 + gear_ratio_score * 0.5

        return FactorResult(score=score, err=None)

