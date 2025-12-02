from typing import Dict
from xtquant import xtdata

from trading.logger import trading_logger
from core.database import get_stock_detail
from utils.stock.format import get_stock_desc

# 股票代码 -> 订阅号
stock_sub_code: Dict[str, int] = {}

def subscribe_stock(code: str, handler):
  """ 订阅股票行情 """
  if code in stock_sub_code:
    # 已订阅
    return

  stock_sub_code[code] = xtdata.subscribe_quote(
    code,
    '1d',
    callback=handler
  )
  detail = get_stock_detail(code)
  trading_logger.debug(f"开始监控持仓：{get_stock_desc(detail)}")

def unsubscribe_stock(code: str):
  """ 取消订阅股票行情 """
  if code not in stock_sub_code:
    # 未订阅
    return
  xtdata.unsubscribe_quote(stock_sub_code[code])
  del stock_sub_code[code]
  detail = get_stock_detail(code)
  trading_logger.debug(f"取消监控持仓：{get_stock_desc(detail)}")

def is_stock_subscribed(code: str) -> bool:
  """ 是否已订阅股票行情 """
  return code in stock_sub_code
