# coding:utf-8
"""
净利润同比增长率因子

使用Z-score归一化：(value - mean) / std
从配置文件读取均值和标准差
"""
from .helpers import FactorCtx
from .helpers.financial_factor_base import FinancialFactorBase
from core.database.financial import get_profit_growth


class ProfitGrowthFactor(FinancialFactorBase):
    """
    净利润同比增长率因子
    
    使用Z-score归一化，从配置文件读取统计信息
    """

    def __init__(self):
        super().__init__('du_profit_rate')
    
    def _get_indicator_value(self, stock_code: str, query_date, ctx: FactorCtx):
        """获取净利润同比增长率指标"""
        return get_profit_growth(stock_code, query_date)

