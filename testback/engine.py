"""
简单回测引擎

功能：
- 管理现金和持仓
- 执行调仓操作
- 记录每日状态
- 计算基础回测指标
"""

from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import numpy as np

from configs.config_loader import StrategyConfig
from core.strategies.sizers import Sizer
from testback.logger import testback_logger


@dataclass
class DailyRecord:
    """每日记录"""
    date: datetime
    cash: float  # 现金
    positions: Dict[str, int]  # 持仓 {stock_code: shares}
    prices: Dict[str, float]  # 当日价格
    market_value: float  # 持仓市值
    total_value: float  # 总资产
    daily_return: float = 0.0  # 当日收益率


class BacktestEngine:
    """简单回测引擎"""

    def __init__(self, config: StrategyConfig):
        """
        初始化回测引擎

        Args:
            config: 策略配置
        """
        self.config = config
        self.initial_capital = config.capital.initial_capital
        self.commission_rate = config.backtest.commission_rate
        self.slippage_rate = config.backtest.slippage_rate
        self.min_commission = config.backtest.min_commission

        # 当前状态
        self.cash = self.initial_capital
        self.positions: Dict[str, int] = {}  # {stock_code: shares}

        # 历史记录
        self.history: List[DailyRecord] = []

        testback_logger.info(
            f"回测引擎初始化完成 - 初始资金: {self.initial_capital:,.0f}, "
            f"佣金率: {self.commission_rate * 10000:.1f}‱, "
            f"滑点: {self.slippage_rate * 1000:.1f}‰"
        )

    def rebalance(
        self,
        target_stocks: List[str],
        date: datetime,
        prices: Dict[str, float]
    ):
        """
        执行调仓操作

        Args:
            target_stocks: 目标持仓列表（已排序的TopN）
            date: 当前日期
            prices: 当前股票价格 {stock_code: price}
        """
        testback_logger.debug(
            f"[{date.strftime('%Y-%m-%d')}] 开始调仓 - "
            f"当前持仓: {len(self.positions)} 只, 目标: {len(target_stocks)} 只"
        )

        # 1. 卖出不在目标列表中的股票
        to_sell = [stock for stock in self.positions.keys() if stock not in target_stocks]
        for stock in to_sell:
            shares = self.positions[stock]
            if stock in prices:
                self._sell(stock, shares, prices[stock], date)
            else:
                testback_logger.warning(
                    f"[{date.strftime('%Y-%m-%d')}] {stock} 无价格数据，无法卖出"
                )

        # 2. 计算目标持仓
        total_value = self._get_total_value(prices)
        allocation = Sizer.allocate(target_stocks, total_value, prices)

        # 3. 执行买卖调整
        for stock in target_stocks:
            target_shares = allocation.get(stock, 0)
            current_shares = self.positions.get(stock, 0)

            if target_shares > current_shares:
                # 买入
                buy_shares = target_shares - current_shares
                if stock in prices:
                    self._buy(stock, buy_shares, prices[stock], date)
            elif target_shares < current_shares:
                # 卖出
                sell_shares = current_shares - target_shares
                if stock in prices:
                    self._sell(stock, sell_shares, prices[stock], date)

        # 4. 记录当日状态
        self._record_daily(date, prices)

    def _buy(self, stock: str, shares: int, price: float, date: datetime):
        """
        买入股票

        Args:
            stock: 股票代码
            shares: 买入股数
            price: 买入价格
            date: 日期
        """
        if shares <= 0:
            return

        # 考虑滑点（买入价格上浮）
        actual_price = price * (1 + self.slippage_rate)

        # 计算总成本
        cost = shares * actual_price

        # 计算佣金
        commission = max(cost * self.commission_rate, self.min_commission)

        # 总花费
        total_cost = cost + commission

        if total_cost > self.cash:
            testback_logger.warning(
                f"[{date.strftime('%Y-%m-%d')}] 资金不足，无法买入 {stock} "
                f"{shares} 股（需要 {total_cost:,.2f}，可用 {self.cash:,.2f}）"
            )
            return

        # 更新现金
        self.cash -= total_cost

        # 更新持仓
        self.positions[stock] = self.positions.get(stock, 0) + shares

        testback_logger.debug(
            f"[{date.strftime('%Y-%m-%d')}] 买入 {stock} "
            f"{shares} 股 @ {actual_price:.2f} = {cost:,.2f} "
            f"(佣金: {commission:.2f})"
        )

    def _sell(self, stock: str, shares: int, price: float, date: datetime):
        """
        卖出股票

        Args:
            stock: 股票代码
            shares: 卖出股数
            price: 卖出价格
            date: 日期
        """
        if shares <= 0:
            return

        current_shares = self.positions.get(stock, 0)
        if current_shares < shares:
            testback_logger.warning(
                f"[{date.strftime('%Y-%m-%d')}] 持仓不足，无法卖出 {stock} "
                f"{shares} 股（持有 {current_shares} 股）"
            )
            shares = current_shares

        if shares == 0:
            return

        # 考虑滑点（卖出价格下降）
        actual_price = price * (1 - self.slippage_rate)

        # 计算收入
        proceeds = shares * actual_price

        # 计算佣金
        commission = max(proceeds * self.commission_rate, self.min_commission)

        # 实际收入
        net_proceeds = proceeds - commission

        # 更新现金
        self.cash += net_proceeds

        # 更新持仓
        self.positions[stock] -= shares
        if self.positions[stock] == 0:
            del self.positions[stock]

        testback_logger.debug(
            f"[{date.strftime('%Y-%m-%d')}] 卖出 {stock} "
            f"{shares} 股 @ {actual_price:.2f} = {proceeds:,.2f} "
            f"(佣金: {commission:.2f})"
        )

    def _get_total_value(self, prices: Dict[str, float]) -> float:
        """计算总资产"""
        market_value = sum(
            shares * prices.get(stock, 0)
            for stock, shares in self.positions.items()
        )
        return self.cash + market_value

    def _get_market_value(self, prices: Dict[str, float]) -> float:
        """计算持仓市值"""
        return sum(
            shares * prices.get(stock, 0)
            for stock, shares in self.positions.items()
        )

    def _record_daily(self, date: datetime, prices: Dict[str, float]):
        """记录每日状态"""
        market_value = self._get_market_value(prices)
        total_value = self.cash + market_value

        # 计算当日收益率
        daily_return = 0.0
        if len(self.history) > 0:
            prev_value = self.history[-1].total_value
            if prev_value > 0:
                daily_return = (total_value - prev_value) / prev_value

        record = DailyRecord(
            date=date,
            cash=self.cash,
            positions=self.positions.copy(),
            prices=prices.copy(),
            market_value=market_value,
            total_value=total_value,
            daily_return=daily_return
        )

        self.history.append(record)

        testback_logger.debug(
            f"[{date.strftime('%Y-%m-%d')}] 资产: {total_value:,.2f} "
            f"(现金: {self.cash:,.2f}, 持仓: {market_value:,.2f}, "
            f"收益率: {daily_return * 100:+.2f}%)"
        )

    def get_metrics(self) -> Dict[str, float]:
        """
        计算回测指标

        Returns:
            {
                'total_return': 总收益率,
                'annualized_return': 年化收益率,
                'sharpe_ratio': 夏普率,
                'max_drawdown': 最大回撤,
                'volatility': 波动率,
                'final_value': 最终市值,
                'total_trades': 总交易次数,
            }
        """
        if len(self.history) == 0:
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'final_value': self.initial_capital,
                'total_trades': 0,
            }

        # 总收益率
        final_value = self.history[-1].total_value
        total_return = (final_value - self.initial_capital) / self.initial_capital

        # 年化收益率
        days = (self.history[-1].date - self.history[0].date).days
        years = max(days / 365.0, 1 / 365.0)  # 至少1天
        annualized_return = (1 + total_return) ** (1 / years) - 1

        # 提取每日收益率
        daily_returns = np.array([record.daily_return for record in self.history[1:]])

        # 波动率（年化）
        if len(daily_returns) > 1:
            volatility = np.std(daily_returns) * np.sqrt(252)  # 假设一年252个交易日
        else:
            volatility = 0.0

        # 夏普率（假设无风险利率为0）
        if volatility > 0:
            sharpe_ratio = annualized_return / volatility
        else:
            sharpe_ratio = 0.0

        # 最大回撤
        max_drawdown = self._calculate_max_drawdown()

        # 统计交易次数（简单统计：每次持仓变化算一次交易）
        total_trades = sum(
            1 for i in range(1, len(self.history))
            if self.history[i].positions != self.history[i - 1].positions
        )

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'final_value': final_value,
            'total_trades': total_trades,
        }

    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        if len(self.history) == 0:
            return 0.0

        values = np.array([record.total_value for record in self.history])
        peaks = np.maximum.accumulate(values)
        drawdowns = (values - peaks) / peaks

        return float(np.min(drawdowns))

    def print_summary(self):
        """打印回测摘要"""
        metrics = self.get_metrics()

        print("\n" + "=" * 60)
        print("回测结果摘要")
        print("=" * 60)
        print(f"初始资金:   {self.initial_capital:>15,.2f} 元")
        print(f"最终市值:   {metrics['final_value']:>15,.2f} 元")
        print(f"总收益率:   {metrics['total_return'] * 100:>14.2f}%")
        print(f"年化收益率: {metrics['annualized_return'] * 100:>14.2f}%")
        print(f"夏普率:     {metrics['sharpe_ratio']:>15.2f}")
        print(f"最大回撤:   {metrics['max_drawdown'] * 100:>14.2f}%")
        print(f"波动率:     {metrics['volatility'] * 100:>14.2f}%")
        print(f"总交易次数: {int(metrics['total_trades']):>15} 次")
        print(f"回测天数:   {len(self.history):>15} 天")
        print("=" * 60 + "\n")
