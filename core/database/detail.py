from typing import Optional
from xtquant import xtdata

from core.logger import core_logger
from .type import StockDetail

cached_stock_details = {}

def get_stock_detail(stock_code: str) -> Optional[StockDetail]:
  """ 获取股票详情 """
  if stock_code in cached_stock_details:
    return cached_stock_details[stock_code]
  try:
    detail = xtdata.get_instrument_detail(stock_code)
    if detail is None:
      core_logger.error(f'股票详情获取失败: {stock_code}')
      return None
    cached_stock_details[stock_code] = detail
    return detail
  except Exception as e:
    core_logger.error(f'股票详情获取失败: {stock_code}, 错误: {e}')
    return None
