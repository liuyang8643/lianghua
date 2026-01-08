from typing import Optional
from xtquant import xtdata

from core.logger import core_logger
from utils.shared_memory import SharedMemoryCache
from .type import StockDetail
from .delist import get_delist_stock_info

# 全局股票详情缓存（共享内存，支持多进程）
_DETAIL_CACHE = SharedMemoryCache('stock_detail')

def init_stock_detail_cache(stock_codes: list[str]):
  """预加载股票详情到共享内存缓存"""
  core_logger.debug(f"预加载 {len(stock_codes)} 只股票的【详情数据】到共享内存...")
  for stock_code in stock_codes:
    get_stock_detail(stock_code)
  core_logger.debug(f"预加载 {len(stock_codes)} 只股票的【详情数据】预加载完成。")

def get_stock_detail(stock_code: str) -> Optional[StockDetail]:
  """获取股票详情（优先从缓存，支持退市股票）
  
  数据来源优先级：
  1. 共享内存缓存（跨进程）
  2. QMT API（在市股票）
  3. AkShare API（退市股票）
  
  Args:
    stock_code: 股票代码（如 600000.SH）
    
  Returns:
    股票详情对象，失败返回 None
  """
  # 缓存命中，直接返回
  if _DETAIL_CACHE.contains(stock_code):
    return _DETAIL_CACHE.get(stock_code)

  # 尝试从 QMT 获取
  try:
    detail = xtdata.get_instrument_detail(stock_code)
    if detail:
      _DETAIL_CACHE.put(stock_code, detail)
      return detail

    # QMT 失败，尝试从 akshare 获取退市股票
    delist_info_dict = get_delist_stock_info()

    if stock_code not in delist_info_dict:
      return None

    # 构造退市股票详情对象
    info = delist_info_dict[stock_code]
    detail: StockDetail = {
      'ExchangeID': 'SSE' if stock_code.endswith('.SH') else 'SZE',
      'InstrumentID': stock_code.split('.')[0],
      'InstrumentName': info.name,
      'ProductID': '',
      'ProductName': '',
      'ProductType': -1,
      'ExchangeCode': stock_code,
      'UniCode': stock_code,
      'CreateDate': info.list_date.strftime('%Y%m%d'),
      'OpenDate': info.list_date.strftime('%Y%m%d'),
      'ExpireDate': info.delist_date.strftime('%Y%m%d'),
      'PreClose': 0.0,
      'SettlementPrice': 0.0,
      'UpStopPrice': 0.0,
      'DownStopPrice': 0.0,
      'FloatVolume': 0.0,
      'TotalVolume': 0.0,
      'PriceTick': 0.01,
      'VolumeMultiple': 1,
      'MainContract': 0,
      'LastVolume': 0,
      'InstrumentStatus': 1,  # 退市状态
    }

    _DETAIL_CACHE.put(stock_code, detail)
    return detail
  except Exception as e:
    core_logger.error(f'QMT 获取失败: {stock_code}, {e}')
