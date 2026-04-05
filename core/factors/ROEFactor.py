# coding:utf-8
"""
ROE因子（净资产收益率）

使用Z-score归一化：(value - mean) / std
从配置文件读取均值和标准差
"""
from .helpers import FactorCtx, FactorResult
from .helpers.financial_factor_base import FinancialFactorBase
from core.database.financial import get_roe


class ROEFactor(FinancialFactorBase):
    """
    ROE因子（净资产收益率）
    
    使用Z-score归一化，从配置文件读取统计信息
    """

    def __init__(self):
        super().__init__('du_return_on_equity')
    
    def _get_indicator_value(self, stock_code: str, query_date, ctx: FactorCtx):
        """获取ROE指标"""
        return get_roe(stock_code, query_date)

