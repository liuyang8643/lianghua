from .helpers import *
from core.database.financial import get_financial_indicators


class FinancialQuality(BaseFactor):
    """
    财务质量因子

    评分维度（总分 0-100）：
    - 盈利能力 (40分): ROE
    - 成长性 (35分): 净利润增长率 + 营收增长率
    - 财务健康 (25分): 毛利率 + 资产负债率
    """

    def calc(self, ctx: FactorCtx) -> FactorResult:
        try:
            query_date = ctx.base_time.date()
            indicators = get_financial_indicators(
                stock_code=ctx.code,
                query_date=query_date,
                indicator_names=[
                    'du_return_on_equity',
                    'du_profit_rate',
                    'inc_revenue_rate',
                    'sales_gross_profit',
                    'gear_ratio',
                ],
                table_name='PershareIndex',
                use_announce_date=True
            )

            roe = indicators.get('du_return_on_equity')
            if roe is None:
                return FactorResult(score=None, err=ValueError(f"ROE数据缺失: {ctx.code}"))

            score = 0.0

            # 1. 盈利能力 (40分): ROE
            if roe > 20:
                score += 40
            elif roe > 15:
                score += 32
            elif roe > 10:
                score += 22
            elif roe > 5:
                score += 12
            elif roe > 0:
                score += 5

            # 2. 成长性 (35分): 净利润增长率 (20) + 营收增长率 (15)
            profit_growth = indicators.get('du_profit_rate')
            if profit_growth is not None:
                if profit_growth > 30:
                    score += 20
                elif profit_growth > 15:
                    score += 14
                elif profit_growth > 0:
                    score += 7

            revenue_growth = indicators.get('inc_revenue_rate')
            if revenue_growth is not None:
                if revenue_growth > 20:
                    score += 15
                elif revenue_growth > 10:
                    score += 10
                elif revenue_growth > 0:
                    score += 5

            # 3. 财务健康 (25分): 毛利率 (15) + 资产负债率 (10)
            gross_profit = indicators.get('sales_gross_profit')
            if gross_profit is not None:
                if gross_profit > 40:
                    score += 15
                elif gross_profit > 25:
                    score += 10
                elif gross_profit > 10:
                    score += 5

            gear_ratio = indicators.get('gear_ratio')
            if gear_ratio is not None:
                if gear_ratio < 40:
                    score += 10
                elif gear_ratio < 60:
                    score += 6
                elif gear_ratio < 80:
                    score += 2

            return FactorResult(
                score=score,
                err=None,
            )
        except Exception as e:
            return FactorResult(score=None, err=e)
