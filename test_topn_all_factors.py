"""测试 TopN 完整选股流程（包含所有因子）"""

from datetime import datetime
from core.database import allow_buy_stock_code_list
from core.strategies.top_n import TopN
from core.strategies._weights import FactorWeights, FactorTemperatures, RANK_N
from core.strategies.sizers.sizer import Sizer


def test_topn_with_all_factors():
    """测试 TopN 完整选股流程"""

    # 1. 获取股票池（取前5只作为测试）
    base_date = datetime(2024, 12, 5)
    stock_list = allow_buy_stock_code_list(base_date)[:5]

    print(f"\n{'='*60}")
    print(f"测试日期: {base_date.strftime('%Y-%m-%d')}")
    print(f"股票池大小: {len(stock_list)}")
    print(f"启用因子: {len(FactorWeights)} 个")
    print(f"{'='*60}\n")

    # 2. 创建 TopN 实例（自动计算因子分数）
    print("正在计算因子分数...")
    top_n = TopN(stock_list=stock_list, base_date=base_date)

    # 3. 打印因子分数摘要
    print("\n因子分数摘要:")
    print(f"{'因子名称':<15} {'均值':>8} {'标准差':>8} {'最小值':>8} {'最大值':>8} {'有效数':>8}")
    print("-" * 60)

    summary = top_n.get_factor_scores_summary()
    for factor_name, stats in summary.items():
        print(
            f"{factor_name:<15} "
            f"{stats['mean']:>8.4f} "
            f"{stats['std']:>8.4f} "
            f"{stats['min']:>8.4f} "
            f"{stats['max']:>8.4f} "
            f"{stats['valid_count']:>8}"
        )

    # 4. 获取 Top N 排名
    print(f"\n正在选股 (Top {RANK_N})...")
    top_stocks = top_n.get_ordered_stocks(
        n=RANK_N,
        weights=FactorWeights,
        temperatures=FactorTemperatures,
        norm_method='rank'  # 使用rank归一化
    )

    # 5. 打印选股结果
    print(f"\n选股结果 (Top {min(RANK_N, len(top_stocks))}):")
    print(f"{'排名':<6} {'股票代码':<12}")
    print("-" * 20)

    for rank, stock_code in enumerate(top_stocks[:20], 1):
        print(f"{rank:<6} {stock_code:<12}")

    # 6. 仓位分配（使用示例价格）
    total_capital = 1_000_000  # 100万本金
    print(f"\n仓位分配 (总资金: {total_capital:,} 元):")

    # 使用示例价格（实际应用中从数据库获取）
    prices = {stock: 20.0 for stock in top_stocks}  # 假设所有股票均价20元

    # 使用 Sizer 计算分配
    allocations = Sizer.allocate_detailed(
        stocks=top_stocks,
        total_capital=total_capital,
        prices=prices
    )

    print(f"{'股票代码':<12} {'股价':>8} {'股数':>8} {'金额':>12}")
    print("-" * 45)

    total_allocated = 0
    for stock_code, allocation in allocations.items():
        if allocation['shares'] > 0:
            print(
                f"{stock_code:<12} "
                f"{allocation['price']:>8.2f} "
                f"{allocation['shares']:>8} "
                f"{allocation['amount']:>12,.2f}"
            )
            total_allocated += allocation['amount']

    print("-" * 45)
    print(f"{'总计':<29} {total_allocated:>12,.2f}")
    print(f"{'剩余资金':<29} {total_capital - total_allocated:>12,.2f}")

    # 7. 打印配置信息
    print(f"\n因子权重配置:")
    for factor_name, weight in FactorWeights.items():
        temp = FactorTemperatures.get(factor_name, 1.0)
        print(f"  {factor_name:<15} 权重={weight:<5.2f} 温度={temp:<5.2f}")

    print(f"\n{'='*60}")
    print("测试完成！")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    test_topn_with_all_factors()
