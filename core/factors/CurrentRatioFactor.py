# coding:utf-8
"""
流动比率因子

使用Z-score归一化：(value - mean) / std
从配置文件读取均值和标准差
需要从Balance表获取数据
"""
from .helpers import FactorCtx
from .helpers.financial_factor_base import FinancialFactorBase
from core.database.financial import get_current_ratio


class CurrentRatioFactor(FinancialFactorBase):
    """
    流动比率因子
    
    使用Z-score归一化，从配置文件读取统计信息
    需要从Balance表获取数据
    """

    def __init__(self):
        super().__init__('current_ratio')
    
    def _get_indicator_value(self, stock_code: str, query_date, ctx: FactorCtx):
        """获取流动比率指标（从Balance表）"""
        return get_current_ratio(stock_code, query_date)

