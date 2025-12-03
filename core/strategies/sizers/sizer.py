from math import ceil, floor
from typing import List, Optional, TypedDict

HAND_SIZE = 100
MIN_BUY_AMOUNT = 10_000.0
MAX_BUY_AMOUNT = 25_000.0
PRESERVE_AMOUNT = 0.0  # 保留金额

class StockInfo(TypedDict):
  code: str
  price: float

class StockAllocation(TypedDict):
  code: str
  count: int
  amount: float

def get_quick_allocation(stock: StockInfo, budget: float, expect_position_count=0) -> Optional[StockAllocation]:
  """ 在预算之内，购买最少数量的股票 """
  hand_price = stock["price"] * HAND_SIZE
  expect_hand = floor(budget / expect_position_count / hand_price) if expect_position_count > 0 else 0
  target_hand = max(ceil(MIN_BUY_AMOUNT / hand_price), expect_hand)
  target_amount = target_hand * hand_price

  if target_amount > budget - PRESERVE_AMOUNT:
    # 预算不足
    return StockAllocation(
      code=stock['code'],
      count=0,
      amount=0
    )

  return StockAllocation(
    code=stock['code'],
    count=target_hand * HAND_SIZE,
    amount=target_amount
  )

class Sizer:
  @property
  def stocks(self):
    return self._stocks

  @property
  def budget(self):
    return self._budget

  @property
  def allocation(self):
    return self._allocation

  @property
  def rest(self):
    return self._rest

  def __init__(self, stocks: List[StockInfo], budget: float):
    self._stocks = stocks
    self._budget = budget
    self._allocation: list[StockAllocation] = []
    self._rest = budget - PRESERVE_AMOUNT

    # 如果股票列表为空，直接返回空分配
    if not stocks:
      return

    target_buy = budget / len(stocks)

    for stock in stocks:
      target_hand = min(
        max(
          # 预期购买手数
          ceil(target_buy / (stock["price"] * HAND_SIZE)),
          # 最小购买手数
          ceil(MIN_BUY_AMOUNT / (stock["price"] * HAND_SIZE)),
        ),
        # 最大购买手数
        floor(MAX_BUY_AMOUNT / (stock["price"] * HAND_SIZE)),
      )
      target_amount = target_hand * HAND_SIZE * stock["price"]

      if target_hand <= 0:
        continue

      if target_amount <= self._rest:
        self._rest -= target_amount
        self._allocation.append(
          {
            "code": stock['code'],
            "count": target_hand * HAND_SIZE,
            "amount": target_amount
          }
        )
      else:
        buy_rest_hand = floor(self._rest / (stock["price"] * HAND_SIZE))
        buy_rest_amount = buy_rest_hand * HAND_SIZE * stock["price"]

        if buy_rest_hand > 0 and buy_rest_amount >= MIN_BUY_AMOUNT:
          self._rest -= buy_rest_amount
          self._allocation.append(
            {
              "code": stock['code'],
              "count": buy_rest_hand * HAND_SIZE,
              "amount": buy_rest_amount
            }
          )

  def get_stock_allocation(self, stock_code: str):
    return next((x for x in self._allocation if x["code"] == stock_code), None)

if __name__ == "__main__":
  # 定义预算和股票列表
  plan_budget = 11_000
  stock_to_buy = [
    StockInfo(code="apple", price=16.07),
    StockInfo(code="banana", price=35.42),
    StockInfo(code="camera", price=16.47),
    StockInfo(code="digital", price=24.17),
    StockInfo(code="elephant", price=12.34),
  ]
  # 示例1: 使用默认预算和股票列表
  result = Sizer(stock_to_buy, plan_budget)
  print(result)
