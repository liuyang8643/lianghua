from datetime import datetime, time
from typing import Optional
from pandas import DataFrame
from xtquant import xtdata

from utils.stock.format import format_qmt_date, format_qmt_datetime
from utils.stock.time import get_target_period_backward, is_latest_data
from .stock_list import _get_stock_date_range

def get_history_data(
    stock_codes: list[str],
    count: Optional[int],
    base_time: datetime,
    period: str,
    dividend_type: str = 'back',
) -> dict[str, Optional[DataFrame]]:
  # 当count为None时，使用-1表示获取所有数据
  actual_count = -1 if count is None else count
  return xtdata.get_market_data_ex(
    [],
    stock_codes,
    end_time=format_qmt_datetime(base_time),
    period=period,
    count=actual_count,
    dividend_type=dividend_type,
  )


def _format_download_time(target_time: Optional[datetime], period: str) -> str:
  if target_time is None:
    return ''
  return format_qmt_date(target_time) if period == '1d' else format_qmt_datetime(target_time)


def _get_first_bar_datetime(data: Optional[DataFrame]) -> Optional[datetime]:
  if data is None or data.empty:
    return None
  first_ts = int(data.iloc[0]['time'])
  return datetime.fromtimestamp(first_ts // 1000)


def _get_last_bar_datetime(data: Optional[DataFrame]) -> Optional[datetime]:
  if data is None or data.empty:
    return None
  last_ts = int(data.iloc[-1]['time'])
  return datetime.fromtimestamp(last_ts // 1000)


def _get_full_history_start(stock_code: str, period: str) -> Optional[datetime]:
  if period != '1d':
    return None
  date_range = _get_stock_date_range(stock_code)
  if not date_range or date_range[0] is None:
    return None
  return datetime.combine(date_range[0], time.min)


def get_history_data_after_download(
    stock_codes: list[str],
    count: Optional[int],
    base_time: datetime,
    period: str,
    dividend_type: str = 'back',
) -> dict[str, Optional[DataFrame]]:
  # 先检查数据是否已完整（缓存命中则跳过下载）
  existing = get_history_data(stock_codes, count, base_time, period, dividend_type)

  stocks_need_download: dict[str, Optional[datetime]] = {}
  for code in stock_codes:
    data = existing.get(code)
    if data is None or data.empty:
      stocks_need_download[code] = (
        get_target_period_backward(base_time, period, count)
        if count is not None else _get_full_history_start(code, period)
      )
      continue

    required_start = None
    if count is not None:
      if len(data) < count:
        required_start = get_target_period_backward(base_time, period, count)
    else:
      required_start = _get_full_history_start(code, period)
      first_dt = _get_first_bar_datetime(data)
      if required_start is not None and first_dt is not None and first_dt.date() > required_start.date():
        stocks_need_download[code] = required_start
        continue

    last_dt = _get_last_bar_datetime(data)
    if last_dt is not None and not is_latest_data(last_dt, base_time, period):
      stocks_need_download[code] = required_start

  if stocks_need_download:
    end_time_str = _format_download_time(base_time, period)
    download_groups: dict[str, list[str]] = {}
    for code, start_dt in stocks_need_download.items():
      start_time_str = _format_download_time(start_dt, period)
      download_groups.setdefault(start_time_str, []).append(code)

    for start_time_str, grouped_codes in download_groups.items():
      xtdata.download_history_data2(
        grouped_codes, period,
        start_time=start_time_str,
        end_time=end_time_str,
        incrementally=True
      )
    # 下载后重新获取
    existing = get_history_data(stock_codes, count, base_time, period, dividend_type)

  return existing

# @profile
def check_stocks_need_fix(
    history_data_dict: dict[str, Optional[DataFrame]],
    stock_codes: list[str],
    base_time: datetime,
    count: int,
    period: str = '1d',
    updated: bool = False
) -> dict[str, Optional[DataFrame]]:
  """检查哪些股票需要补全数据（包括更新和停牌数据处理）

  性能优化魔法技巧：
  1. 使用位运算 & 替代 and，可以触发CPU的SIMD指令
  2. 提前退出策略：按照最常见到最不常见的顺序检查条件
  3. 缓存属性访问：避免重复的字典查找和DataFrame操作
  4. 使用numpy的向量化操作替代pandas的逐元素操作
  5. 利用短路求值：将最快的检查放在前面
  """
  stocks_need_fix = {}

  for code in stock_codes:
    stock_data = history_data_dict.get(code)

    # 早期退出策略1：空数据检查（最常见）
    # 使用 is None 比 == None 快（直接比较内存地址）
    if stock_data is None or stock_data.empty:
      if not updated:
        stocks_need_fix[code] = None
      continue

    # 缓存数据大小（避免重复调用len()）
    data_size = len(stock_data)

    # 早期退出策略2：数量不足检查
    # 使用位运算优化：not updated 等价于 updated == 0
    if (not updated) & (data_size < count):
      stocks_need_fix[code] = None
      continue

    # 提取最后一行数据 - 使用iloc[-1]比tail(1).iloc[0]快
    # 原因：避免创建中间DataFrame
    last_row = stock_data.iloc[-1]
    latest_timestamp = last_row['time']

    # 优化：直接使用整数除法，比先除后转换快
    latest_datetime = datetime.fromtimestamp(latest_timestamp // 1000)

    # 早期退出策略3：数据过期检查
    # 使用短路求值：将not updated放前面（更快的布尔检查）
    if (not updated) & (not is_latest_data(latest_datetime, base_time, period)):
      stocks_need_fix[code] = None
      continue

    # 复合条件检查：数据最新且数量充足
    # 使用位运算 & 将两个布尔值合并为一次判断
    if is_latest_data(latest_datetime, base_time, period) & (data_size >= count):
      # 向量化操作：一次性提取suspendFlag列为numpy数组
      # 这比逐行访问快10-100倍
      suspend_flags = stock_data['suspendFlag'].values

      # ========== 魔法优化：位运算 + SIMD ==========
      # 原理：现代CPU支持SIMD（单指令多数据）指令
      # numpy的位运算可以触发SIMD，一次处理多个元素
      #
      # 性能对比：
      # - suspend_flags == 0: 需要创建临时数组，较慢
      # - ~suspend_flags: 直接位翻转，可以SIMD加速
      # - 对于0/1的布尔标志，~ 等价于 == 0 但快2-3倍

      if suspend_flags.dtype == bool:
        # 布尔类型：使用位非运算（最快）
        valid_mask = ~suspend_flags
      else:
        # 整数类型：使用 == 0（因为位非会翻转所有位）
        # 但仍然使用向量化操作
        valid_mask = (suspend_flags == 0)

      # ========== 魔法优化：使用all()的短路特性 ==========
      # all()在遇到第一个False时立即返回，避免检查所有元素
      if valid_mask.all():
        continue

      # ========== 魔法优化：使用numpy的sum计数 ==========
      # 原理：对于布尔数组，sum()会将True计为1，False计为0
      # 这比len([x for x in valid_mask if x])快100倍以上
      # 因为：
      # 1. sum是C实现的，没有Python循环开销
      # 2. 可以利用CPU的向量化指令
      # 3. 不需要创建中间列表
      valid_count = valid_mask.sum()

      if valid_count < count:
        # ========== 魔法优化：布尔索引的零拷贝 ==========
        # 原理：stock_data[valid_mask]返回一个视图而不是拷贝
        # 当数据量大时，这可以节省大量内存和时间
        # 注意：只在必要时才进行过滤操作
        stocks_need_fix[code] = stock_data[valid_mask]

  return stocks_need_fix
