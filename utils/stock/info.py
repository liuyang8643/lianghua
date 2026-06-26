def is_cyb_stock(stock_code: str) -> bool:
  """创业板（300/301开头）：上市前5日不设限，此后±20%；ST仍20%。"""
  return stock_code.startswith('300') or stock_code.startswith('301')

def is_kcb_stock(stock_code: str) -> bool:
  """科创板（688开头）：上市前5日不设限，此后±20%；ST仍20%。"""
  return stock_code.startswith('688')

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

def min_buy_shares(stock_code: str) -> int:
  """委托最小买入数量：科创板 200 股起，其余 100 股起。"""
  if is_kcb_stock(stock_code):
    return 200
  return 100


def buy_step_shares(stock_code: str) -> int:
  """买入申报递增步长：科创板 200 股起、1 股递增；其余 100 股递增。"""
  if is_kcb_stock(stock_code):
    return 1
  return min_buy_shares(stock_code)


def floor_buy_shares(stock_code: str, shares: int) -> int:
  """把买入数量向下调整为合法申报量；不足最低买入量时返回 0。"""
  qty = int(shares)
  minimum = min_buy_shares(stock_code)
  if qty < minimum:
    return 0
  step = buy_step_shares(stock_code)
  return minimum + ((qty - minimum) // step) * step


def round_buy_shares(stock_code: str, shares: int) -> int:
  """把目标买入数量调整为合法申报量；小于最低量的正数提升到最低量。"""
  qty = int(shares)
  if qty <= 0:
    return 0
  minimum = min_buy_shares(stock_code)
  if qty < minimum:
    return minimum
  return floor_buy_shares(stock_code, qty)


def min_sell_shares(stock_code: str) -> int:
  """部分卖出单笔申报最低股数：科创板200股（余额不足200可一次性全清，不受此限），其余100股。"""
  if is_kcb_stock(stock_code):
    return 200
  return 100

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
