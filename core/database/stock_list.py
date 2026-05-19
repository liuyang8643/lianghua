from datetime import datetime, date
from functools import lru_cache
from typing import Optional, Union

from core.logger import core_logger
from utils.stock.info import is_b_stock
from .detail import get_stock_detail
from .delist import get_delist_stock_info


@lru_cache(maxsize=1)
def _fetch_all_a_stocks() -> tuple[str, ...]:
  """获取全部A股股票代码（通过xtdata本地数据）"""
  from xtquant import xtdata

  codes = xtdata.get_stock_list_in_sector('沪深A股')
  core_logger.debug(f"xtdata A股全部股票: {len(codes)} 只")
  return tuple(sorted(codes))


@lru_cache(maxsize=6000)
def _get_stock_date_range(stock_code: str) -> Optional[tuple[date, Optional[date]]]:
  """获取股票有效日期范围: (上市日期, 退市日期)，退市日期为None表示未退市"""
  # 优先使用 akshare 退市数据，避免对退市股票调用接口
  delist_info = get_delist_stock_info()
  if stock_code in delist_info:
    info = delist_info[stock_code]
    return info.list_date, info.delist_date

  # 非退市股票
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


def get_all_stock_code_list(target_date: Optional[Union[datetime, date]] = None) -> list[str]:
  """获取所有A股股票列表（不含B股）

  Args:
    target_date: 可选，指定日期。支持 datetime 或 date 类型。如果传入则按日期过滤有效股票，否则返回所有股票
  """
  # 统一转换为 date
  if target_date is not None:
    if isinstance(target_date, datetime):
      target_date = target_date.date()

  stocks = set(_fetch_all_a_stocks())

  # 补充akshare退市股票
  delist_info = get_delist_stock_info()
  stocks.update(delist_info.keys())

  # 排除B股
  filtered = {
    code for code in stocks
    if not is_b_stock(code)
  }

  # 根据日期过滤（如果提供了日期）
  if target_date:
    valid = [code for code in filtered if check_stock_valid_at_date(code, target_date)]
    result = tuple(sorted(valid))
    core_logger.debug(f"获取 {target_date.strftime('%Y-%m-%d')} 有效股票: {len(stocks)} -> {len(filtered)}(排除B股) -> {len(result)}(日期过滤)")
  else:
    result = tuple(sorted(filtered))
    core_logger.debug(f"获取所有股票: {len(stocks)} -> {len(filtered)}(排除B股)")

  return list(result)


def allow_buy_stock_code_list(
    target_date: Optional[Union[datetime, date]] = None,
    boards: Optional[tuple[str, ...]] = None,
) -> list[str]:
  """获取可买入股票池（ST 过滤由因子控制）

  Args:
    target_date: 不传=全量含退市, 传入=排除该日期前已退市但保留之后退市的
    boards: 板块前缀过滤, 如 ('60','00') 只保留主板
  """
  result = get_all_stock_code_list(target_date)
  if boards:
    result = [s for s in result if s.startswith(boards)]

  if not result:
    core_logger.error('可买股票池为空')
    return []

  extra = ''
  if target_date:
    if isinstance(target_date, datetime):
      target_date = target_date.date()
    extra += f' date={target_date.isoformat()}'
  if boards:
    extra += f' boards={boards}'
  core_logger.info(f'可买股票池{extra}: {len(result)} 只')
  return result
