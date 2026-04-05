# coding:utf-8
"""
营收同比增长率因子

使用Z-score归一化：(value - mean) / std
从配置文件读取均值和标准差
"""
from .helpers import FactorCtx
from .helpers.financial_factor_base import FinancialFactorBase
from core.database.financial import get_revenue_growth


class RevenueGrowthFactor(FinancialFactorBase):
    """
    营收同比增长率因子
    
    使用Z-score归一化，从配置文件读取统计信息
    """

    def __init__(self):
        super().__init__('inc_revenue_rate')
    
    def _get_indicator_value(self, stock_code: str, query_date, ctx: FactorCtx):
        """获取营收同比增长率指标"""
        return get_revenue_growth(stock_code, query_date)

