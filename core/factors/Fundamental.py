"""
基本面因子 - 基于 AkShare 财务数据

设计特点:
1. 数据获取集成在因子内部（无需单独数据层）
2. 利用类变量缓存全市场数据（避免重复 API 调用）
3. 复用 @cached_factor 机制（按股票缓存结果）
"""
import akshare as ak
import pandas as pd
from datetime import datetime
from typing import Optional, TYPE_CHECKING
from core.factors.helpers.interface import BaseFactor, FactorResult
from core.factors.helpers.cache import cached_factor

if TYPE_CHECKING:
    from core.factors.helpers.indicators import FactorCtx


class Fundamental(BaseFactor):
    """
    基本面综合因子

    评分逻辑（加法混合）:
    - ROE > 15%: +0.25
    - 利润增长 > 20%: +0.30
    - 营收增长 > 15%: +0.20
    - 毛利率 > 30%: +0.15
    - 现金流 > 0: +0.10

    总分范围: 0-1（正向因子，分数越高越好）
    """

    # 类变量：缓存全市场数据（避免重复 API 调用）
    _market_cache = {}  # {report_period: DataFrame}

    @cached_factor("Fundamental")
    def calc(self, ctx: 'FactorCtx') -> FactorResult:
        """
        计算基本面因子

        由于 @cached_factor 装饰器，每只股票每天只会计算一次
        缓存路径: {stock_code}/{stock_code}_{date}_factor_Fundamental_{hash}.pkl
        """
        # 获取财务数据
        financial = self._get_financial(ctx.code, ctx.base_time)

        if financial is None:
            return FactorResult(score=0.5, err="无财务数据")

        # 计算分数
        score = 0.0

        # 1. ROE 评分 (0.25)
        roe = financial.get('roe')
        if pd.notna(roe):
            if roe > 15:
                score += 0.25
            elif roe > 10:
                score += 0.15
            elif roe > 5:
                score += 0.08

        # 2. 利润增长评分 (0.30)
        profit_growth = financial.get('profit_growth')
        if pd.notna(profit_growth):
            if profit_growth > 20:
                score += 0.30
            elif profit_growth > 10:
                score += 0.18
            elif profit_growth > 0:
                score += 0.08

        # 3. 营收增长评分 (0.20)
        revenue_growth = financial.get('revenue_growth')
        if pd.notna(revenue_growth):
            if revenue_growth > 15:
                score += 0.20
            elif revenue_growth > 5:
                score += 0.12
            elif revenue_growth > 0:
                score += 0.05

        # 4. 毛利率评分 (0.15)
        gross_margin = financial.get('gross_margin')
        if pd.notna(gross_margin):
            if gross_margin > 30:
                score += 0.15
            elif gross_margin > 20:
                score += 0.10
            elif gross_margin > 10:
                score += 0.05

        # 5. 现金流评分 (0.10)
        cash_flow = financial.get('cash_flow_ps')
        if pd.notna(cash_flow) and cash_flow > 0:
            score += 0.10

        return FactorResult(score=score)

    def _get_financial(self, stock_code: str, date: datetime) -> Optional[pd.Series]:
        """
        获取财务数据（利用类变量缓存）

        策略:
        1. 第一只股票查询时，调用 API 获取全市场数据，存入类变量
        2. 后续股票直接从类变量读取
        3. 每个报告期只调用一次 API
        """
        # 确定报告期
        report_period = self._get_latest_quarter(date)

        # 检查类变量缓存
        if report_period not in self._market_cache:
            # 首次查询：调用 API 获取全市场数据
            try:
                df = ak.stock_yjbb_em(date=report_period)
                self._market_cache[report_period] = df
            except Exception as e:
                print(f"获取财务数据失败 (报告期: {report_period}): {e}")
                return None

        # 从缓存中提取当前股票
        df = self._market_cache[report_period]
        clean_code = stock_code[:6]  # 去掉 .SH/.SZ 后缀
        stock_data = df[df['股票代码'] == clean_code]

        if stock_data.empty:
            return None

        # 提取关键字段
        row = stock_data.iloc[0]
        return pd.Series({
            'eps': row.get('每股收益'),
            'bps': row.get('每股净资产'),
            'roe': row.get('净资产收益率'),
            'revenue_growth': row.get('营业总收入-同比增长'),
            'profit_growth': row.get('净利润-同比增长'),
            'cash_flow_ps': row.get('每股经营现金流量'),
            'gross_margin': row.get('销售毛利率'),
        })

    def _get_latest_quarter(self, date: datetime) -> str:
        """
        根据日期推测最新可用的财报期（考虑发布延迟）

        假设财报发布延迟 3-4 个月:
        - 11-12 月 -> 使用 Q3 (0930)
        - 8-10 月  -> 使用 Q2 (0630)
        - 5-7 月   -> 使用 Q1 (0331)
        - 1-4 月   -> 使用上一年年报 (1231)
        """
        year, month = date.year, date.month

        if month >= 11:
            return f"{year}0930"
        elif month >= 8:
            return f"{year}0630"
        elif month >= 5:
            return f"{year}0331"
        else:
            return f"{year-1}1231"

    @classmethod
    def clear_market_cache(cls):
        """清除类变量缓存（用于测试或切换报告期）"""
        cls._market_cache.clear()
