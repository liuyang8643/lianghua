from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
  from core.database import StockDetail, StockTradingData


def get_stock_estimate_up_limit(stock_code: str, current_price: float) -> float:
  if is_st_stock(stock_code):
    return current_price * 1.049  # ST ±5%
  up_limit_rate = 1.199 if is_cyb_stock(stock_code) else 1.099
  return current_price * up_limit_rate

def get_stock_estimate_down_limit(stock_code: str, current_price: float) -> float:
  if is_st_stock(stock_code):
    return current_price * 0.951  # ST ±5%
  down_limit_rate = 0.801 if is_cyb_stock(stock_code) else 0.901
  return current_price * down_limit_rate

def is_stock_trading(detail: Optional['StockDetail']) -> bool:
  """ 判断股票是否正常交易 """

  if detail is None:
    return False

  return (
      detail['InstrumentStatus'] <= 0  # 未停牌
      # 未退市
      and (detail['ExpireDate'] in ('0', '99999999'))
  )

def is_st_stock(stock_code: str) -> bool:
  """ 判断股票名称是否含 ST（*ST / ST 等） """
  from core.database import get_stock_detail
  detail = get_stock_detail(stock_code)
  return bool(detail and 'ST' in detail.get('InstrumentName', ''))

def is_cyb_stock(stock_code: str) -> bool:
  """ 判断股票是否为创业板股票 """
  return stock_code.startswith('300') or stock_code.startswith('301')

def is_kcb_stock(stock_code: str) -> bool:
  """ 判断股票是否为科创板股票 """
  return stock_code.startswith('688')

def is_bse_stock(stock_code: str) -> bool:
  """ 判断股票是否为北交所股票 """
  return stock_code.startswith('43') or stock_code.startswith('83')

def is_b_stock(stock_code: str) -> bool:
  """ 判断股票是否为B股 """
  return stock_code.startswith('900') or stock_code.startswith('200')

def is_convertible_bond(stock_code: str) -> bool:
  """ 判断股票是否为可转债 """
  return stock_code.startswith('11') or stock_code.startswith('12') or stock_code.startswith('13')

# baseline_stock_code = '000300.SH'  # 基准股票代码，沪深300
baseline_stock_code = '000852.SH'  # 基准股票代码，中证1000

def get_baseline_data(base_time: datetime = None) -> Optional['StockTradingData']:
  """ 获取基准价格 """
  from core.database import get_market_data_from_cache

  input_time = base_time or datetime.now()
  data = get_market_data_from_cache(baseline_stock_code, 1, input_time, '1d')
  return data.iloc[-1] if data is not None and data.size > 0 else None
