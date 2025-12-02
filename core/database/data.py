# from line_profiler_pycharm import profile
import pandas as pd
from datetime import datetime
from typing import Optional
import numpy as np

from utils.stock.format import format_qmt_datetime
from utils.stock.time import get_latest_trading_time, is_latest_data
from .history import check_stocks_need_fix, get_history_data, get_history_data_after_download

# 全局历史数据缓存，以股票代码为key，存储完整历史数据
_GLOBAL_DAILY_CACHE: dict[str, Optional[pd.DataFrame]] = {}
_GLOBAL_MINUTE_CACHE: dict[str, Optional[pd.DataFrame]] = {}

def clear_market_data_cache(code: str = None, period: str = None):
  """清除市场数据缓存

  Args:
    code: 股票代码，如果为None则清除所有
    period: 数据周期 ('1d' 或 '1m')，如果为None则清除所有周期
  """
  if code:
    if period is None or period == '1d':
      _GLOBAL_DAILY_CACHE.pop(code, None)
    if period is None or period == '1m':
      _GLOBAL_MINUTE_CACHE.pop(code, None)
  else:
    if period is None or period == '1d':
      _GLOBAL_DAILY_CACHE.clear()
    if period is None or period == '1m':
      _GLOBAL_MINUTE_CACHE.clear()

def get_full_market_data(stock_code: str, period: str = '1d', target_time: datetime = None) -> Optional[pd.DataFrame]:
  """获取股票的完整历史数据（从上市日到目标时间），使用全局缓存

  性能优化：
  - 使用全局缓存避免重复IO
  - 对于日线数据一次性加载完整历史
  """
  if target_time is None:
    target_time = datetime.now()

  cache = _GLOBAL_DAILY_CACHE if period == '1d' else _GLOBAL_MINUTE_CACHE

  # 如果缓存中已有数据，直接返回（零拷贝）
  if stock_code in cache and cache[stock_code] is not None:
    return cache[stock_code]

  # 缓存中没有，需要加载数据
  if period == '1d':
    # 日线数据：不传入count参数，获取所有可用数据
    data = get_market_data(stock_code, None, target_time, '1d')
  else:
    # 分钟数据：默认获取最近30个交易日（约7500条）
    data = get_market_data(stock_code, 7500, target_time, '1m', allow_tainted=True)

  # 缓存数据
  cache[stock_code] = data

  return data

def get_market_data_from_cache(
    stock_code: str,
    count: int,
    base_time: datetime,
    period: str = '1d',
) -> Optional[pd.DataFrame]:
  """从缓存中获取指定时间范围的市场数据

  性能优化：
  - 使用numpy的searchsorted进行二分查找（O(log n)复杂度）
  - 使用iloc切片实现零拷贝视图
  - 避免重复的timestamp转换
  """
  # 获取完整历史数据
  full_data = get_full_market_data(stock_code, period)

  if full_data is None or full_data.empty:
    return None

  # 根据base_time筛选出<=base_time的数据
  # 优化：使用二分查找，因为time列是升序排列的
  base_time_ms = int(base_time.timestamp() * 1000)

  # 使用 searchsorted 进行二分查找，找到第一个 > base_time_ms 的位置
  time_values = full_data['time'].values
  insert_pos = np.searchsorted(time_values, base_time_ms, side='right')

  # 切分数据：取 [:insert_pos] 就是所有 <= base_time_ms 的数据
  if insert_pos < count:
    # 数据不足，尝试重新加载更多数据（对于分钟数据可能需要更多）
    if period == '1m':
      required_size = count + 2000
      _GLOBAL_MINUTE_CACHE[stock_code] = None  # 清除旧缓存
      # 重新加载更多数据
      full_data = get_market_data(stock_code, required_size, base_time, '1m', allow_tainted=True)
      _GLOBAL_MINUTE_CACHE[stock_code] = full_data

      if full_data is not None and not full_data.empty:
        time_values = full_data['time'].values
        insert_pos = np.searchsorted(time_values, base_time_ms, side='right')

    if insert_pos < count:
      return None

  # 使用 iloc 直接切片，比 tail() 更高效（零拷贝视图）
  start_pos = max(0, insert_pos - count)
  return full_data.iloc[start_pos:insert_pos]

def get_market_data(
    stock_code: str,
    count: Optional[int],
    base_time: datetime = None,
    period: str = '1d',
    allow_tainted: bool = False,  # 是否允许返回不完整或非最新的数据
) -> Optional[pd.DataFrame]:
  """ deprecated, use get_market_data_batch instead """
  history_data = get_market_data_batch([stock_code], count, base_time, period, allow_tainted)[stock_code]
  # 当count为None时，不校验数据数量
  if not allow_tainted and count is not None:
    if history_data is None or history_data['time'].size < count:
      raise ValueError(f'{stock_code} 获取 {format_qmt_datetime(base_time)} {count}*{period} 失败：数据不足')
    latest_date = datetime.fromtimestamp((history_data.iloc[-1]['time']) / 1000)
    if not is_latest_data(latest_date, base_time, period):
      raise ValueError(f'{stock_code} 获取 {format_qmt_datetime(base_time)} {count}*{period} 失败：最新数据为 {format_qmt_datetime(latest_date)}')
  return history_data

# @profile
def get_market_data_batch(
    stock_codes: list[str],
    count: Optional[int],
    base_time: datetime = None,
    period: str = '1d',
    allow_tainted: bool = False,  # 是否允许返回不完整或非最新的数据
) -> dict[str, Optional[pd.DataFrame]]:
  """批量获取市场数据

  性能优化：
  - 批量IO操作减少网络往返
  - 快速路径：如果允许污染数据直接返回
  - 优化停牌数据处理，减少重复操作
  """
  if not stock_codes:
    return {}

  input_time = base_time or datetime.now()
  latest_trading_time = get_latest_trading_time(input_time)

  history_data_dict = get_history_data(stock_codes, count, latest_trading_time, period)

  # 如果允许污染数据，直接返回，跳过所有校验和修复逻辑
  if allow_tainted:
    return history_data_dict

  # 当count为None时，仍然需要检查并修复过期数据，但最终允许返回不完整数据
  skip_count_validation = (count is None)

  # 检查股票需要修复（更新或停牌数据处理）
  for attempt in range(3):
    stocks_need_fix = check_stocks_need_fix(
      history_data_dict,
      stock_codes,
      input_time,
      count if count is not None else 10000,  # count为None时使用一个大数值避免数量检查
      period,
      attempt > 0
    )

    if not stocks_need_fix:
      break

    # 批量修复股票数据
    tainted_data: dict[str, Optional[pd.DataFrame]] = {}
    stocks_need_update: list[str] = []
    stocks_need_suspend_fix: list[str] = []

    for code, valid_data in stocks_need_fix.items():
      tainted_data[code] = history_data_dict[code]
      history_data_dict[code] = None
      if valid_data is None:
        stocks_need_update.append(code)
      else:
        stocks_need_suspend_fix.append(code)

    # 批量更新过期数据
    if stocks_need_update:
      updated_dict = get_history_data_after_download(stocks_need_update, count, latest_trading_time, period)

      for code, updated_data in updated_dict.items():
        if updated_data is not None and not updated_data.empty:
          # 当count为None时，返回所有数据；否则取最后count条
          if skip_count_validation:
            history_data_dict[code] = updated_data
          else:
            history_data_dict[code] = updated_data.iloc[-count:] if len(updated_data) > count else updated_data

    # 批量处理停牌数据补全（仅当count不为None时才处理）
    if stocks_need_suspend_fix and not skip_count_validation:
      # 向量化计算批量请求参数
      earliest_times = [datetime.fromtimestamp(tainted_data[c].iloc[0]['time'] / 1000) for c in stocks_need_suspend_fix]
      max_earliest_time = max(earliest_times)

      data_sizes = [stocks_need_fix[c]['time'].size for c in stocks_need_suspend_fix]
      estimated_count = max(count - size for size in data_sizes)

      more_data_dict = get_history_data_after_download(stocks_need_suspend_fix, estimated_count * (attempt + 1) + 1, max_earliest_time, period)

      # 处理返回数据 - 优化版本：使用向量化过滤
      filtered_more_data = {}
      for code, prepend_data in more_data_dict.items():
        if prepend_data is not None and not prepend_data.empty:
          # 使用numpy数组过滤，比pandas更快
          suspend_flags = prepend_data['suspendFlag'].values
          valid_mask = (suspend_flags == 0)
          if valid_mask.any():
            # 只在有有效数据时才创建新DataFrame
            filtered_more_data[code] = prepend_data[valid_mask]

      # 批量合并和去重
      for code in stocks_need_suspend_fix:
        if code in filtered_more_data and not filtered_more_data[code].empty:
          valid_data = stocks_need_fix[code]
          prepend_data = filtered_more_data[code]

          # 优化合并策略：只有当预添加数据不为空时才进行合并
          if len(prepend_data) > 0:
            # 使用更高效的合并方式：pd.concat + drop_duplicates
            combined_data = pd.concat([prepend_data, valid_data], ignore_index=True)
            # 使用numpy进行去重判断会更快
            time_values = combined_data['time'].values
            _, unique_indices = np.unique(time_values, return_index=True)
            unique_indices.sort()  # 保持时间顺序
            combined_data = combined_data.iloc[unique_indices]

            # 使用iloc切片取最后count条
            history_data_dict[code] = combined_data.iloc[-count:] if len(combined_data) > count else combined_data
          else:
            # 如果没有预添加数据，直接使用原有数据
            suspend_mask = valid_data['suspendFlag'].values == 0
            filtered = valid_data[suspend_mask]
            history_data_dict[code] = filtered.iloc[-count:] if len(filtered) > count else filtered
        else:
          # 如果没有找到更多数据，使用原有数据
          valid_data = stocks_need_fix[code]
          suspend_mask = valid_data['suspendFlag'].values == 0
          filtered = valid_data[suspend_mask]
          history_data_dict[code] = filtered.iloc[-count:] if len(filtered) > count else filtered

  return history_data_dict
