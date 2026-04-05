# coding:utf-8
"""
资产负债率因子（转换为正向因子）

使用Z-score归一化：(value - mean) / std
从配置文件读取均值和标准差

设计理念：资产负债率越低越好，转换为正向指标
转换方式：使用 100 - gear_ratio，使得值越大越好
"""
from .helpers import FactorCtx
from .helpers.financial_factor_base import FinancialFactorBase
from core.database.financial import get_gear_ratio


class GearRatioFactor(FinancialFactorBase):
    """
    资产负债率因子（正向）
    
    使用Z-score归一化，从配置文件读取统计信息
    在因子设计时将资产负债率转换为正向指标：100 - gear_ratio
    这样转换后的值越大，表示财务越稳健
    """

    def __init__(self):
        super().__init__('gear_ratio')
    
    def _get_indicator_value(self, stock_code: str, query_date, ctx: FactorCtx):
        """
        获取资产负债率指标并转换为正向指标
        
        转换公式：100 - gear_ratio
        这样资产负债率越低，转换后的值越大，因子得分越高
        """
        gear_ratio = get_gear_ratio(stock_code, query_date)
        if gear_ratio is None:
            return None
        # 转换为正向指标：100 - 资产负债率
        # 资产负债率越低，转换后的值越大，因子得分越高
        return 100.0 - gear_ratio

