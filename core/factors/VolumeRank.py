"""
成交量排名百分比因子

计算最近N个交易日的成交量排名百分比：
- 排名第1（成交量最大）= 100%
- 排名第N（成交量最小）= 0%

设计逻辑：
- 成交量排名越高（成交量越大），分数越高
- 用于识别放量突破或异常放量
"""

import numpy as np
from .helpers import BaseFactor, FactorResult, FactorCtx


class VolumeRank(BaseFactor):
    """
    成交量排名百分比因子
    
    计算方式：
    1. 获取最近N个交易日的成交量数据
    2. 计算当前成交量在这N天中的排名（从大到小）
    3. 转换为百分比：排名1 = 100%，排名N = 0%
    
    输出：原始分数 [0, 1]，由框架层进行batch norm归一化
    """

    def __init__(self, period: int = 100):
        """
        :param period: 计算周期，默认100个交易日
        """
        super().__init__()
        self.period = period

    def calc(self, ctx: FactorCtx) -> FactorResult:
        # 获取最近N个交易日的日线数据
        history_data = ctx.get_daily_data(self.period)
        
        if history_data is None:
            raise ValueError(f"无法获取历史数据: {ctx.code}")
        
        if len(history_data) < self.period:
            raise ValueError(f"历史数据不足，需要至少{self.period}个交易日，实际只有{len(history_data)}个: {ctx.code}")
        
        # 提取成交量数据
        volumes = history_data['volume'].values
        
        if len(volumes) < self.period:
            raise ValueError(f"成交量数据不足，需要{self.period}个，实际只有{len(volumes)}个: {ctx.code}")
        
        # 检查是否有无效值
        if np.any(np.isnan(volumes)):
            raise ValueError(f"成交量数据包含NaN值: {ctx.code}")
        
        if np.any(volumes <= 0):
            raise ValueError(f"成交量数据包含非正值: {ctx.code}")
        
        # 获取当前成交量（最后一个）
        current_volume = float(volumes[-1])
        
        if np.isnan(current_volume):
            raise ValueError(f"当前成交量数据为NaN: {ctx.code}")
        
        if current_volume <= 0:
            raise ValueError(f"当前成交量数据无效（<=0）: {ctx.code}, volume={current_volume}")
        
        # 计算排名（从大到小排序，排名1是最大的）
        # 方法：计算有多少个成交量严格大于当前成交量，然后+1得到排名
        # 例如：如果有0个大于当前，排名=1；如果有99个大于当前，排名=100
        greater_count = np.sum(volumes > current_volume)
        rank = greater_count + 1  # 排名从1开始
        
        # 转换为百分比：排名1 = 100%，排名N = 0%
        # 线性映射公式：score = (N - rank) / (N - 1)
        # rank=1时：score = (N-1)/(N-1) = 1.0 = 100%
        # rank=N时：score = (N-N)/(N-1) = 0.0 = 0%
        if self.period == 1:
            # 特殊情况：只有1天数据
            score = 1.0
        else:
            score = (self.period - rank) / (self.period - 1)
        
        return FactorResult(
            score=score,
            err=None,
            metadata={
                'rank': rank,
                'current_volume': current_volume,
                'period': self.period
            }
        )

