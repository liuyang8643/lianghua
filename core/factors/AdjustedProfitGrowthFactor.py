# coding:utf-8
"""
扣非净利润同比增长率因子

使用Z-score归一化：(value - mean) / std
从配置文件读取均值和标准差
"""
from .helpers.financial_factor_base import FinancialIndicatorFactor


class AdjustedProfitGrowthFactor(FinancialIndicatorFactor):
    """
    扣非净利润同比增长率因子
    
    使用Z-score归一化，从配置文件读取统计信息
    """

    def __init__(self):
        super().__init__('adjusted_net_profit_rate', 'adjusted_net_profit_rate', 'PershareIndex')

