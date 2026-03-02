"""退市股票数据管理（统一数据源）"""
from datetime import datetime, date
from functools import lru_cache
from typing import NamedTuple

from core.logger import core_logger

class DelistStockInfo(NamedTuple):
  """退市股票信息"""
  name: str  # 股票简称
  list_date: date  # 上市日期
  delist_date: date  # 退市日期

@lru_cache(maxsize=1)
def get_delist_stock_info() -> dict[str, DelistStockInfo]:
  """从 akshare 获取退市股票信息（跨进程磁盘缓存 + 进程内 lru_cache，每日刷新）
  
  Returns:
    {股票代码: DelistStockInfo(name, list_date, delist_date)}
  """
  # L2: 跨进程磁盘缓存（每日 key，次日自动失效）
  from core.factors.helpers.cache import DiskCache, CacheKey
  cache_key = CacheKey.make_key(['delist', date.today().strftime('%Y%m%d')])
  cached = DiskCache.load_pickle(cache_key)
  if cached is not None:
    return cached

  delist_info = {}

  try:
    import akshare as ak

    # 上交所退市股票
    try:
      df = ak.stock_info_sh_delist()
      for _, row in df.iterrows():
        code = f"{row['公司代码']}.SH"
        name = str(row.get('公司简称', ''))
        list_date_str = str(row.get('上市日期', ''))
        delist_date_str = str(row.get('暂停上市日期', ''))

        # 验证必需字段
        if not name or name == 'nan' or not list_date_str or list_date_str == 'nan':
          continue
        if not delist_date_str or delist_date_str == 'nan':
          continue

        # 解析日期
        try:
          list_date = datetime.strptime(list_date_str, '%Y-%m-%d').date()
          delist_date = datetime.strptime(delist_date_str, '%Y-%m-%d').date()
          delist_info[code] = DelistStockInfo(name, list_date, delist_date)
        except ValueError:
          continue
    except Exception as e:
      core_logger.warning(f'获取上交所退市股票失败: {e}')

    # 深交所退市股票
    try:
      df = ak.stock_info_sz_delist()
      for _, row in df.iterrows():
        code = f"{row['证券代码']}.SZ"
        name = str(row.get('证券简称', ''))
        list_date_str = str(row.get('上市日期', ''))
        delist_date_str = str(row.get('终止上市日期', ''))

        # 验证必需字段
        if not name or name == 'nan' or not list_date_str or list_date_str == 'nan':
          continue
        if not delist_date_str or delist_date_str == 'nan':
          continue

        # 解析日期
        try:
          list_date = datetime.strptime(list_date_str, '%Y-%m-%d').date()
          delist_date = datetime.strptime(delist_date_str, '%Y-%m-%d').date()
          delist_info[code] = DelistStockInfo(name, list_date, delist_date)
        except ValueError:
          continue
    except Exception as e:
      core_logger.warning(f'获取深交所退市股票失败: {e}')
  except ImportError:
    core_logger.warning('未安装 akshare，无法获取退市股票信息')
  except Exception as e:
    core_logger.error(f'获取退市股票信息异常: {e}')

  DiskCache.save_pickle(cache_key, delist_info)
  return delist_info
