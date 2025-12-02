from datetime import date, datetime
from typing import Optional

from core.database import StockDetail

def get_stock_code(stock_detail: StockDetail) -> str:
  return f"{stock_detail['ExchangeCode']}.{stock_detail['ExchangeID']}" if stock_detail else '-'

def get_stock_desc(stock_detail: StockDetail) -> str:
  return f"【{get_stock_code(stock_detail)}】{stock_detail['InstrumentName']}" if stock_detail else '-'

def get_rate(target_price: float, base_price: float) -> Optional[float]:
  """
  获取价格比较率
  :param target_price: 目标价格
  :param base_price: 基准价格
  :return: 价格比较率
  """
  if target_price is None or base_price is None:
    return None
  return (target_price - base_price) / base_price

def get_rate_fmt(target_price: float, base_price: float) -> str:
  value = get_rate(target_price, base_price)
  if value is None:
    return "N/A"
  return f"{value:.2%}"

def format_qmt_date(base_date: date = None):
  date_ins = base_date or date.today()
  return date_ins.strftime('%Y%m%d')

def format_qmt_datetime(base_time: datetime = None):
  date_time_ins = base_time or datetime.now()
  return date_time_ins.strftime('%Y%m%d%H%M%S')
