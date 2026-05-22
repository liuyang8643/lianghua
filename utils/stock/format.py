from __future__ import annotations
from datetime import date, datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
  from data.db.type import StockDetail

def get_stock_code(stock_detail: StockDetail) -> str:
  return f"{stock_detail['ExchangeCode']}.{stock_detail['ExchangeID']}" if stock_detail else '-'

def get_stock_desc(stock_detail: StockDetail) -> str:
  return f"【{get_stock_code(stock_detail)}】{stock_detail['InstrumentName']}" if stock_detail else '-'

def format_qmt_date(base_date: date = None):
  date_ins = base_date or date.today()
  return date_ins.strftime('%Y%m%d')

def format_qmt_datetime(base_time: datetime = None):
  date_time_ins = base_time or datetime.now()
  return date_time_ins.strftime('%Y%m%d%H%M%S')
