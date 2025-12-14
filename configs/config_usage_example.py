"""配置系统使用示例

演示如何使用 StrategyConfig 加载和使用策略配置
"""

from configs.config_loader import StrategyConfig


def example_load_config():
    """示例1: 加载策略配置"""
    print("=" * 60)
    print("示例1: 加载策略配置")
    print("=" * 60)

    # 加载策略配置（自动继承 base.yaml）
    config = StrategyConfig.load("strategy_v1.yaml")

    # 访问配置
    print(f"\n策略信息:")
    print(f"  名称: {config.meta.name}")
    print(f"  版本: {config.meta.version}")
    print(f"  描述: {config.meta.description}")

    print(f"\n选股配置:")
    print(f"  选股数量: {config.selection.rank_n}")

    print(f"\n资金配置:")
    print(f"  初始资金: {config.capital.initial_capital:,}")
    print(f"  最大持仓比例: {config.capital.max_position_pct}%")


def example_get_factors():
    """示例2: 获取启用的因子"""
    print("\n" + "=" * 60)
    print("示例2: 获取启用的因子")
    print("=" * 60)

    config = StrategyConfig.load("strategy_v1.yaml")

    # 获取启用的因子
    enabled_factors = config.get_enabled_factors()
    print(f"\n启用因子: {len(enabled_factors)} 个")

    for factor in enabled_factors:
        print(f"\n  {factor.name}:")
        print(f"    权重: {factor.weight}")
        print(f"    温度: {factor.temperature}")
        if factor.params:
            print(f"    参数: {factor.params}")


def example_get_weights():
    """示例3: 获取因子权重和温度"""
    print("\n" + "=" * 60)
    print("示例3: 获取因子权重和温度")
    print("=" * 60)

    config = StrategyConfig.load("strategy_v1.yaml")

    # 获取权重字典
    weights = config.get_factor_weights()
    print("\n因子权重:")
    for name, weight in weights.items():
        print(f"  {name}: {weight}")

    # 获取温度字典
    temperatures = config.get_factor_temperatures()
    print("\n因子温度:")
    for name, temp in temperatures.items():
        print(f"  {name}: {temp}")


def example_compare_configs():
    """示例4: 对比不同策略配置"""
    print("\n" + "=" * 60)
    print("示例4: 对比不同策略配置")
    print("=" * 60)

    v1 = StrategyConfig.load("strategy_v1.yaml")
    v2 = StrategyConfig.load("strategy_v2.yaml")

    print(f"\n策略对比:")
    print(f"{'配置项':<20} {'V1':<15} {'V2':<15}")
    print("-" * 50)
    print(f"{'选股数量':<20} {v1.selection.rank_n:<15} {v2.selection.rank_n:<15}")
    print(f"{'初始资金':<20} {v1.capital.initial_capital:<15,} {v2.capital.initial_capital:<15,}")

    print(f"\n因子权重对比:")
    v1_weights = v1.get_factor_weights()
    v2_weights = v2.get_factor_weights()

    all_factors = set(v1_weights.keys()) | set(v2_weights.keys())
    for factor in sorted(all_factors):
        w1 = v1_weights.get(factor, 0.0)
        w2 = v2_weights.get(factor, 0.0)
        print(f"  {factor:<15} {w1:<10.2f} {w2:<10.2f}")


def example_use_in_backtest():
    """示例5: 在回测中使用配置"""
    print("\n" + "=" * 60)
    print("示例5: 在回测中使用配置")
    print("=" * 60)

    config = StrategyConfig.load("current.yaml")

    # 模拟回测初始化
    print(f"\n初始化回测:")
    print(f"  策略: {config.meta.name}")
    print(f"  选股数: {config.selection.rank_n}")
    print(f"  初始资金: {config.capital.initial_capital:,}")

    # 获取因子配置
    weights = config.get_factor_weights()
    temperatures = config.get_factor_temperatures()

    print(f"\n因子配置:")
    for name in weights.keys():
        print(f"  {name}: 权重={weights[name]}, 温度={temperatures[name]}")

    # 获取回测参数
    print(f"\n回测参数:")
    print(f"  开始日期: {config.backtest.start_date}")
    print(f"  结束日期: {config.backtest.end_date}")
    print(f"  佣金率: {config.backtest.commission}%")
    print(f"  滑点: {config.backtest.slippage}%")


if __name__ == "__main__":
    # 运行所有示例
    example_load_config()
    example_get_factors()
    example_get_weights()
    example_compare_configs()
    example_use_in_backtest()

    print("\n" + "=" * 60)
    print("示例演示完成！")
    print("=" * 60)
