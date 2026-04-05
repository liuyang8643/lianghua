# coding:utf-8
"""
毛利率因子

使用Z-score归一化：(value - mean) / std
从配置文件读取均值和标准差
"""
from .helpers.financial_factor_base import FinancialIndicatorFactor


class GrossProfitFactor(FinancialIndicatorFactor):
    """
    毛利率因子
    
    使用Z-score归一化，从配置文件读取统计信息
    """

    def __init__(self):
        super().__init__('gross_profit', 'gross_profit', 'PershareIndex')

