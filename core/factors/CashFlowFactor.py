# coding:utf-8
"""
每股经营活动现金流量因子

使用Z-score归一化：(value - mean) / std
从配置文件读取均值和标准差
"""
from .helpers import FactorCtx
from .helpers.financial_factor_base import FinancialFactorBase
from core.database.financial import get_cash_flow_ps


class CashFlowFactor(FinancialFactorBase):
    """
    每股经营活动现金流量因子
    
    使用Z-score归一化，从配置文件读取统计信息
    """

    def __init__(self):
        super().__init__('s_fa_ocfps')
    
    def _get_indicator_value(self, stock_code: str, query_date, ctx: FactorCtx):
        """获取每股经营活动现金流量指标"""
        return get_cash_flow_ps(stock_code, query_date)

