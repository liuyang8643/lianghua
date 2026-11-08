def is_cyb_stock(stock_code: str) -> bool:
  """创业板（300/301开头）：上市前5日不设限，此后±20%；ST仍20%。"""
  return stock_code.startswith('300') or stock_code.startswith('301')

def is_bse_stock(stock_code: str) -> bool:
  """北交所（43/83/87/92开头）：上市首日不设限，此后±30%。"""
  bare_code = stock_code.split('.')[0]
  return bare_code.startswith(('43', '83', '87', '92'))

def is_b_stock(stock_code: str) -> bool:
  """B股（900/200开头）"""
  return stock_code.startswith('900') or stock_code.startswith('200')

def is_convertible_bond(stock_code: str) -> bool:
  """可转债（11/12/13开头）"""
  return stock_code.startswith('11') or stock_code.startswith('12') or stock_code.startswith('13')

from utils.stock.quantity import (
    is_kcb_stock, minimum_buy_quantity as min_buy_shares,
    buy_quantity_step as buy_step_shares, floor_buy_quantity as floor_buy_shares,
    floor_partial_sell_quantity,
)


def round_buy_shares(stock_code: str, shares: int) -> int:
  """把目标买入数量调整为合法申报量；小于最低量的正数提升到最低量。"""
  qty = int(shares)
  if qty <= 0:
    return 0
  minimum = min_buy_shares(stock_code)
  if qty < minimum:
    return minimum
  return floor_buy_shares(stock_code, qty)


min_sell_shares = min_buy_shares


def floor_sell_shares(stock_code: str, requested: int, available: int, holding: int) -> int:
  """Execution quantity: cap by holdings/availability; permit odd-lot full liquidation."""
  quantity = min(int(requested), int(available), int(holding))
  if quantity <= 0:
    return 0
  if quantity == int(holding):
    return quantity
  return floor_partial_sell_quantity(stock_code, quantity)

def board_limit_ratio(stock_code: str) -> float:
  """板块常规涨跌幅比例：科创/创业板0.20，北交所0.30，主板0.10。"""
  if is_kcb_stock(stock_code) or is_cyb_stock(stock_code):
    return 0.20
  if is_bse_stock(stock_code):
    return 0.30
  return 0.10

def limit_up_price(stock_code: str, prev_close: float) -> float:
  """涨停价 = 前收 × (1 + 板块涨跌幅)，用于市价买单资金冻结估算。"""
  if not prev_close or prev_close <= 0:
    return 0.0
  return float(prev_close) * (1.0 + board_limit_ratio(stock_code))
