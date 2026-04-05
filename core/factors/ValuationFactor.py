# coding:utf-8
"""
估值因子（方案B - 因子4，可选）

输入指标：
- PE（市盈率）= 股价 / EPS
- PB（市净率）= 股价 / BPS
- PEG = PE / 净利润增长率

输出：估值分数（连续分数，正向因子，低估值高分）
使用分位数截断+Sigmoid连续映射，确保每个指标贡献范围一致
"""
from .helpers import *
import numpy as np
from core.database.financial import (
    get_eps,
    get_bps,
    get_profit_growth
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


class ValuationFactor(BaseFactor):
    """
    估值因子

    使用连续反向Sigmoid函数评分（低估值 = 高分，基于合理估计范围）：
    - PE：1%分位=5, 99%分位=100, 中心点=20, 权重40%（反向，越低越好）
    - PB：1%分位=0.5, 99%分位=10, 中心点=2, 权重30%（反向，越低越好）
    - PEG：1%分位=0.1, 99%分位=10, 中心点=1, 权重30%（反向，越低越好）

    每个指标先映射到0-1范围，然后加权平均，确保贡献一致
    """

    def __init__(self):
        super().__init__()

    def calc(self, ctx: FactorCtx) -> FactorResult:
        query_date = ctx.base_time.date()
        stock_code = ctx.code

        # 获取财务指标
        eps = get_eps(stock_code, query_date)
        bps = get_bps(stock_code, query_date)
        profit_growth = get_profit_growth(stock_code, query_date)

        if eps is None:
            raise ValueError(f"无EPS数据: {stock_code}")
        if bps is None:
            raise ValueError(f"无BPS数据: {stock_code}")
        if profit_growth is None:
            raise ValueError(f"无净利润增长率数据: {stock_code}")

        # 获取当前股价
        current_price = ctx.get_current_price()
        if current_price is None or current_price <= 0:
            raise ValueError(f"无有效股价数据: {stock_code}")

        # 计算估值指标
        if eps <= 0:
            raise ValueError(f"EPS无效（<=0）: {stock_code}, eps={eps}")
        if bps <= 0:
            raise ValueError(f"BPS无效（<=0）: {stock_code}, bps={bps}")
        
        pe = current_price / eps
        pb = current_price / bps
        peg = pe / profit_growth if profit_growth > 0 else float('inf')

        # 使用连续函数计算每个指标的分数（都映射到0-1范围）
        # 如果值为inf，使用最差分数（对于反向指标，inf对应最低分）
        pe_score = continuous_score(pe, p1=5, p99=100, center=20, scale=8, reverse=True) if pe != float('inf') else 0.0
        pb_score = continuous_score(pb, p1=0.5, p99=10, center=2, scale=8, reverse=True) if pb != float('inf') else 0.0
        peg_score = continuous_score(peg, p1=0.1, p99=10, center=1, scale=8, reverse=True) if peg != float('inf') else 0.0

        # 加权平均（每个指标贡献范围都是0-1，不会因为范围差距而丢失信息）
        score = pe_score * 0.4 + pb_score * 0.3 + peg_score * 0.3

        return FactorResult(score=score, err=None)

