"""
仓位管理工具 - 简化版等额分配模式

计算逻辑：
- 总资金 / 股票数量 = 每只股票分配金额
- 每只股票买入数量 = (分配金额 / 股价) // 100 * 100 （向下取整到100的倍数）
"""

from typing import Dict, List


class Sizer:
    """简化的仓位管理器 - 等额分配模式"""

    HAND_SIZE = 100  # 一手股数

    @staticmethod
    def allocate(
        stocks: List[str],
        total_capital: float,
        prices: Dict[str, float],
        hand_size: int = 100
    ) -> Dict[str, int]:
        """
        等额分配资金并计算每只股票的买入数量

        Args:
            stocks: 股票代码列表
            total_capital: 总资金
            prices: 股票价格字典 {stock_code: price}
            hand_size: 一手股数（默认100）

        Returns:
            {stock_code: shares} - 每只股票的买入股数
        """
        if len(stocks) == 0:
            return {}

        # 等额分配
        amount_per_stock = total_capital / len(stocks)

        allocation = {}

        for stock in stocks:
            if stock not in prices:
                # 没有价格数据的股票，跳过
                allocation[stock] = 0
                continue

            price = prices[stock]

            if price <= 0:
                # 价格无效，跳过
                allocation[stock] = 0
                continue

            # 计算能买多少手（向下取整）
            hands = int(amount_per_stock / price / hand_size)

            # 转换为股数
            shares = hands * hand_size

            allocation[stock] = shares

        return allocation

    @staticmethod
    def allocate_detailed(
        stocks: List[str],
        total_capital: float,
        prices: Dict[str, float],
        hand_size: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """
        等额分配（返回详细信息）

        Args:
            stocks: 股票代码列表
            total_capital: 总资金
            prices: 股票价格字典
            hand_size: 一手股数

        Returns:
            {
                stock_code: {
                    'shares': 股数,
                    'hands': 手数,
                    'amount': 实际金额,
                    'allocated_amount': 分配金额,
                    'price': 股价,
                }
            }
        """
        if len(stocks) == 0:
            return {}

        amount_per_stock = total_capital / len(stocks)

        allocation = {}

        for stock in stocks:
            if stock not in prices:
                allocation[stock] = {
                    'shares': 0,
                    'hands': 0,
                    'amount': 0.0,
                    'allocated_amount': amount_per_stock,
                    'price': 0.0,
                }
                continue

            price = prices[stock]

            if price <= 0:
                allocation[stock] = {
                    'shares': 0,
                    'hands': 0,
                    'amount': 0.0,
                    'allocated_amount': amount_per_stock,
                    'price': price,
                }
                continue

            # 计算手数
            hands = int(amount_per_stock / price / hand_size)
            shares = hands * hand_size
            actual_amount = shares * price

            allocation[stock] = {
                'shares': shares,
                'hands': hands,
                'amount': actual_amount,
                'allocated_amount': amount_per_stock,
                'price': price,
            }

        return allocation

    @staticmethod
    def get_total_usage(allocation: Dict[str, int], prices: Dict[str, float]) -> Dict[str, float]:
        """
        计算资金使用情况

        Args:
            allocation: 分配结果 {stock_code: shares}
            prices: 股票价格 {stock_code: price}

        Returns:
            {
                'total_cost': 总花费,
                'stock_count': 持仓股票数,
                'share_count': 总股数,
            }
        """
        total_cost = 0.0
        stock_count = 0
        share_count = 0

        for stock, shares in allocation.items():
            if shares > 0 and stock in prices:
                cost = shares * prices[stock]
                total_cost += cost
                stock_count += 1
                share_count += shares

        return {
            'total_cost': total_cost,
            'stock_count': stock_count,
            'share_count': share_count,
        }


# 测试和示例
if __name__ == "__main__":
    # 测试数据
    test_stocks = [
        '000001.SZ',
        '000002.SZ',
        '600000.SH',
        '600036.SH',
        '601318.SH',
    ]

    test_prices = {
        '000001.SZ': 10.50,
        '000002.SZ': 25.80,
        '600000.SH': 8.90,
        '600036.SH': 35.20,
        '601318.SH': 42.50,
    }

    test_capital = 100_000  # 10万资金

    print("=" * 60)
    print("简化仓位管理器测试")
    print("=" * 60)
    print(f"总资金: {test_capital:,.0f} 元")
    print(f"股票数量: {len(test_stocks)}")
    print(f"每只分配: {test_capital / len(test_stocks):,.0f} 元")
    print()

    # 简单分配
    print("1. 简单分配结果:")
    allocation = Sizer.allocate(test_stocks, test_capital, test_prices)
    for stock, shares in allocation.items():
        if shares > 0:
            price = test_prices[stock]
            amount = shares * price
            print(f"  {stock}: {shares:>5} 股 ({shares // 100:>3} 手) "
                  f"= {amount:>10,.2f} 元 @ {price:.2f}")
    print()

    # 详细分配
    print("2. 详细分配结果:")
    detailed = Sizer.allocate_detailed(test_stocks, test_capital, test_prices)
    for stock, info in detailed.items():
        print(f"  {stock}:")
        print(f"    分配金额: {info['allocated_amount']:>10,.2f} 元")
        print(f"    股价: {info['price']:>10,.2f} 元")
        print(f"    买入: {info['shares']:>5} 股 ({info['hands']:>3} 手)")
        print(f"    实际花费: {info['amount']:>10,.2f} 元")
    print()

    # 资金使用情况
    print("3. 资金使用情况:")
    usage = Sizer.get_total_usage(allocation, test_prices)
    print(f"  总花费: {usage['total_cost']:>10,.2f} 元")
    print(f"  剩余: {test_capital - usage['total_cost']:>10,.2f} 元")
    print(f"  使用率: {usage['total_cost'] / test_capital * 100:>6.2f}%")
    print(f"  持仓股票数: {usage['stock_count']}")
    print(f"  总股数: {usage['share_count']:,}")
