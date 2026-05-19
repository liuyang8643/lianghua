from typing import Optional

from core.logger import core_logger
from utils.shared_memory import SharedMemoryCache
from .type import StockDetail
from .delist import get_delist_stock_info

_DETAIL_CACHE = SharedMemoryCache('stock_detail')


def _fetch_stock_detail_xtdata(stock_code: str) -> Optional[StockDetail]:
  """通过 xtdata 本地数据获取股票详情"""
  from xtquant import xtdata

  detail = xtdata.get_instrument_detail(stock_code)
  if not detail or not detail.get('InstrumentName'):
    return None

  exchange = detail.get('ExchangeID', '')
  if exchange == 'SH':
    exchange = 'SSE'
  elif exchange == 'SZ':
    exchange = 'SZE'
  else:
    return None

  return {
    'ExchangeID': exchange,
    'InstrumentID': detail.get('InstrumentID', ''),
    'InstrumentName': detail.get('InstrumentName', ''),
    'ProductID': '',
    'ProductName': '',
    'ProductType': -1,
    'ExchangeCode': stock_code,
    'UniCode': stock_code,
    'CreateDate': detail.get('CreateDate', 0),
    'OpenDate': detail.get('OpenDate', ''),
    'ExpireDate': detail.get('ExpireDate', '99999999'),
    'PreClose': 0.0,
    'SettlementPrice': 0.0,
    'UpStopPrice': 0.0,
    'DownStopPrice': 0.0,
    'FloatVolume': float(detail.get('FloatVolume', 0) or 0),
    'TotalVolume': float(detail.get('TotalVolume', 0) or 0),
    'PriceTick': 0.01,
    'VolumeMultiple': 1,
    'MainContract': 0,
    'LastVolume': 0,
    'InstrumentStatus': 0,
  }


def init_stock_detail_cache(stock_codes: list[str]):
  """预加载股票详情到共享内存缓存"""
  core_logger.debug(f"预加载 {len(stock_codes)} 只股票的【详情数据】到共享内存...")
  for stock_code in stock_codes:
    get_stock_detail(stock_code)
  core_logger.debug(f"预加载 {len(stock_codes)} 只股票的【详情数据】预加载完成。")
  stat = _DETAIL_CACHE.stat()
  core_logger.debug(f"_DETAIL_CACHE 包含 {stat['count']} 条数据，共计 {stat['total_size_mb']:.2f} MB。")


def get_stock_detail(stock_code: str) -> Optional[StockDetail]:
  """获取股票详情（优先缓存，通过 xtdata 本地数据获取）"""
  if _DETAIL_CACHE.contains(stock_code):
    return _DETAIL_CACHE.get(stock_code)

  try:
    detail = _fetch_stock_detail_xtdata(stock_code)
    if detail:
      _DETAIL_CACHE.put(stock_code, detail)
      return detail

    delist_info_dict = get_delist_stock_info()
    if stock_code not in delist_info_dict:
      return None

    info = delist_info_dict[stock_code]
    bare_code = stock_code.split('.')[0]
    exchange = 'SSE' if stock_code.endswith('.SH') else ('BSE' if stock_code.endswith('.BJ') else 'SZE')
    detail: StockDetail = {
      'ExchangeID': exchange,
      'InstrumentID': bare_code,
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
      'InstrumentStatus': 1,
    }
    _DETAIL_CACHE.put(stock_code, detail)
    return detail
  except Exception as e:
    core_logger.error(f'获取详情失败: {stock_code}, {e}')
    return None
