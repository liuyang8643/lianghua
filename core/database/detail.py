from datetime import datetime
from typing import Optional
from xtquant import xtdata

from core.logger import core_logger
from utils.shared_memory import SharedMemoryCache
from utils.stock.format import format_qmt_datetime
from .type import StockDetail

# 全局股票详情缓存，使用共享内存支持多进程
_GLOBAL_STOCK_DETAIL_CACHE = SharedMemoryCache('stock_detail')

def init_stock_detail_cache(stock_codes: list[str]):
  """初始化股票详情缓存，预加载指定股票列表的详情数据"""
  for stock_code in stock_codes:
    get_stock_detail(stock_code)

def get_stock_detail(stock_code: str) -> Optional[StockDetail]:
  """获取股票详情（使用全局缓存）
  
  性能优化：
  - 使用全局共享内存缓存避免重复API调用
  - 支持多进程间数据共享
  - 自动压缩存储节省内存
  
  Args:
    stock_code: 股票代码
    
  Returns:
    股票详情对象，失败返回 None
  """
  # 如果缓存中已有数据，直接返回（零拷贝）
  if _GLOBAL_STOCK_DETAIL_CACHE.contains(stock_code):
    cached = _GLOBAL_STOCK_DETAIL_CACHE.get(stock_code)
    if cached is not None:
      return cached

  # 缓存未命中，从 QMT API 获取
  try:
    detail = xtdata.get_instrument_detail(stock_code)
    if detail is None:
      core_logger.error(f'股票详情获取失败: {stock_code}')
      return None

    # 缓存数据
    _GLOBAL_STOCK_DETAIL_CACHE.put(stock_code, detail)
    return detail

  except Exception as e:
    core_logger.error(f'股票详情获取失败: {stock_code}, 错误: {e}')
    return None

def check_stock_date_valid(stock_code: str, target_date: datetime, count: int, period: str):
  """检查股票在指定时间点是否有效（存在且未退市）"""
  detail = get_stock_detail(stock_code)
  if detail is None:
    raise ValueError(f'{stock_code} 获取 {format_qmt_datetime(target_date)} {count}*{period} 失败：股票不存在')
  if detail['OpenDate'] in ('00000000', ''):
    raise ValueError(f'{stock_code} 获取 {format_qmt_datetime(target_date)} {count}*{period} 失败：股票未上市')
  if detail['OpenDate'] is not None and detail['ExpireDate'] is not None:
    try:
      open_date = datetime.strptime(detail['OpenDate'], '%Y%m%d')
      if target_date < open_date:
        raise ValueError(f'{stock_code} 获取 {format_qmt_datetime(target_date)} {count}*{period} 失败：请求时间早于上市时间 {format_qmt_datetime(open_date)}')
    except Exception as e:
      # core_logger.warning(f'{stock_code} OpenDate 解析失败: {detail["OpenDate"]}, 错误: {e}')
      pass
    try:
      expire_date = datetime.strptime(detail['ExpireDate'], '%Y%m%d') if detail['ExpireDate'] not in ('0', '99999999') else None
      if expire_date is not None and target_date > expire_date:
        raise ValueError(f'{stock_code} 获取 {format_qmt_datetime(target_date)} {count}*{period} 失败：请求时间晚于退市时间 {format_qmt_datetime(expire_date)}')
    except Exception as e:
      # core_logger.warning(f'{stock_code} ExpireDate 解析失败: {detail["ExpireDate"]}, 错误: {e}')
      pass
