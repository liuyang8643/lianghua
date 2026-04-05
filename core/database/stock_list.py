from datetime import datetime, date
from functools import lru_cache
from typing import Optional, Union
from xtquant import xtdata

from core.logger import core_logger
from utils.stock.info import is_kcb_stock, is_bse_stock, is_b_stock
from .detail import get_stock_detail
from .delist import get_delist_stock_info

_last_sector_update_date: date | None = None

def _ensure_sector_data_updated():
  """确保板块数据已更新（每天只更新一次）"""
  global _last_sector_update_date
  if _last_sector_update_date != date.today():
    xtdata.download_history_contracts()
    core_logger.debug(f'更新板块数据成功：{len(xtdata.get_sector_list())} 条')
    _last_sector_update_date = date.today()

@lru_cache(maxsize=6000)
def _get_stock_date_range(stock_code: str) -> Optional[tuple[date, Optional[date]]]:
  """获取股票有效日期范围: (上市日期, 退市日期)，退市日期为None表示未退市"""
  # 优先使用 akshare 退市数据，避免对退市股票调用 QMT 接口
  delist_info = get_delist_stock_info()
  if stock_code in delist_info:
    info = delist_info[stock_code]
    return info.list_date, info.delist_date

  # 非退市股票调用QMT接口
  detail = get_stock_detail(stock_code)
  if not detail:
    return None

  open_date_str = detail.get('OpenDate')
  if not open_date_str or open_date_str == '00000000':
    return None

  try:
    return datetime.strptime(open_date_str, '%Y%m%d').date(), None
  except ValueError:
    core_logger.warning(f'{stock_code} 上市日期格式错误: {open_date_str}')
    return None

def check_stock_valid_at_date(stock_code: str, target_date: date) -> bool:
  """检查股票在指定日期是否有效
  
  Args:
    stock_code: 股票代码
    target_date: 目标日期
  """
  date_range = _get_stock_date_range(stock_code)
  if not date_range:
    return False

  open_date, expire_date = date_range
  if target_date < open_date:
    return False
  if expire_date and target_date > expire_date:
    return False
  return True

@lru_cache(maxsize=128)
def _get_sector_stocks(sector_name: str) -> tuple[str, ...]:
  """获取指定板块的股票代码"""
  _ensure_sector_data_updated()
  try:
    stock_list = xtdata.get_stock_list_in_sector(sector_name)
    if stock_list:
      core_logger.debug(f"板块 [{sector_name}] 包含 {len(stock_list)} 只股票")
      return tuple(stock_list)
    return tuple()
  except Exception as e:
    core_logger.error(f"获取板块 [{sector_name}] 股票列表失败: {e}")
    return tuple()

def get_all_stock_code_list(target_date: Optional[Union[datetime, date]] = None) -> list[str]:
  """获取所有A股股票列表（不含科创板和北交所）

  Args:
    target_date: 可选，指定日期。支持 datetime 或 date 类型。如果传入则按日期过滤有效股票，否则返回所有股票
  """
  # 统一转换为 date
  if target_date is not None:
    if isinstance(target_date, datetime):
      target_date = target_date.date()

  stocks = set()

  # 获取QMT在市股票（不含B股，科创板后续过滤）
  for sector in ["沪深A股", "沪深风险警示", "沪深退市整理"]:
    stocks.update(_get_sector_stocks(sector))
  qmt_count = len(stocks)

  # 补充akshare退市股票
  delist_info = get_delist_stock_info()
  before_count = len(stocks)
  stocks.update(delist_info.keys())
  added = len(stocks) - before_count

  core_logger.debug(f'股票池: QMT在市 {qmt_count} 只 + AkShare退市 {added} 只 = 总计 {len(stocks)} 只')

  # 排除科创板、北交所和B股
  filtered = {
    code for code in stocks
    if not (is_kcb_stock(code) or is_bse_stock(code) or is_b_stock(code))
  }

  # 根据日期过滤（如果提供了日期）
  if target_date:
    valid = [code for code in filtered if check_stock_valid_at_date(code, target_date)]
    result = tuple(sorted(valid))
    core_logger.debug(f"获取 {target_date.strftime('%Y-%m-%d')} 有效股票: {len(stocks)} -> {len(filtered)}(排除科创/北交/B股) -> {len(result)}(日期过滤)")
  else:
    result = tuple(sorted(filtered))
    core_logger.debug(f"获取所有股票: {len(stocks)} -> {len(filtered)}(排除科创/北交/B股)")

  return list(result)


def allow_buy_stock_code_list() -> list[str]:
  """获取当前可买入的股票列表（排除科创板、北交所、退市股；ST 过滤由因子控制）"""
  all_a = set(_get_sector_stocks("沪深A股"))

  exclude = set()
  for sector in ["科创板"]:
    exclude.update(_get_sector_stocks(sector))

  result = list(all_a - exclude)

  if not result:
    core_logger.error('获取可买股票失败，返回空列表')
    return []

  core_logger.info(f'可买股票筛选完成：{len(all_a)} -> {len(result)} 只（排除科创板）')
  return result
