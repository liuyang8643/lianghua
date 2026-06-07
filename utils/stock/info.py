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
  """市价委托最小买入数量：科创/创业板200股，主板/北交所100股。"""
  if is_kcb_stock(stock_code) or is_cyb_stock(stock_code):
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
