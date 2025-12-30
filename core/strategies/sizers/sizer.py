PRESERVE_AMOUNT = 0.0  # 保留金额

from typing import Dict, List, Tuple

class Sizer:
  """
  简化的仓位管理器 - 等额分配模式

  计算逻辑：
  - 总资金 / 股票数量 = 每只股票分配金额
  - 每只股票买入数量 = (分配金额 / 股价) // 100 * 100 （向下取整到100的倍数）
  """

  @staticmethod
  def allocate(
      stock_infos: List[Tuple[str, float]],
      total_capital: float,
      hand_size: int = 100
  ) -> Dict[str, int]:
    """
    等额分配资金并计算每只股票的买入数量

    Args:
        stock_infos: 股票信息列表 [(stock_code, price), ...]
        total_capital: 总资金
        hand_size: 一手股数（默认100）

    Returns:
        {stock_code: shares} - 每只股票的买入股数
    """
    if len(stock_infos) == 0:
      return {}

    # 等额分配
    amount_per_stock = (total_capital - PRESERVE_AMOUNT) / len(stock_infos)

    allocation = {}

    for code, price in stock_infos:
      if not price or price <= 0:
        # 没有价格数据的股票，跳过
        allocation[code] = 0
        continue

      # 计算能买多少手（向下取整）
      hands = int(amount_per_stock / price / hand_size)

      # 转换为股数
      allocation[code] = hands * hand_size

    return allocation

# 测试和示例
if __name__ == "__main__":
  # 测试数据
  test_stock_infos = [
    ('000001.SZ', 10.50),
    ('000002.SZ', 25.80),
    ('600000.SH', 8.90),
    ('600036.SH', 35.20),
    ('601318.SH', 42.50),
  ]

  test_capital = 100_000  # 10万资金

  # 简单分配
  print("1. 简单分配结果:")
  a = Sizer.allocate(test_stock_infos, test_capital)
  for stock, shares in a.items():
    print(f"{stock} × {shares}) ")
