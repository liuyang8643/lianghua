# from line_profiler_pycharm import profile
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import current_process
import pandas as pd
from datetime import date, datetime
from typing import Optional
import numpy as np

from core.logger import core_logger
from utils.shared_memory import SharedMemoryCache
from utils.stock.format import format_qmt_datetime
from utils.stock.time import get_latest_trading_time, get_trading_date_span, is_latest_data
from .history import check_stocks_need_fix, get_history_data, get_history_data_after_download
from .stock_list import check_stock_valid_at_date

# 全局市场数据缓存，以股票代码+复权方式为 key，按需存储共享窗口数据
_GLOBAL_DAILY_CACHE = SharedMemoryCache('daily')
_GLOBAL_MINUTE_CACHE = SharedMemoryCache('minute')



def _make_market_cache_key(stock_code: str, dividend_type: str) -> str:
  return f'{stock_code}:{dividend_type}'


def _get_market_cache(period: str) -> SharedMemoryCache:
  return _GLOBAL_DAILY_CACHE if period == '1d' else _GLOBAL_MINUTE_CACHE


def _filter_active_bars(data: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
  if data is not None and not data.empty and 'suspendFlag' in data.columns:
    # 使用 numpy 向量化操作，比 pandas boolean indexing 更快
    suspend_mask = data['suspendFlag'].values == 0
    data = data[suspend_mask]
  return data


def init_full_data(stock_codes: list[str] = None, period: str = '1d', max_workers: int = None, dividend_type: str = 'back'):
  """兼容旧接口；市场数据已改为按需窗口加载。"""
  core_logger.debug(
    "init_full_data 已不再执行全量历史预加载；请改用按需缓存或 init_market_data_range()."
  )
  return 0


def _cache_market_data_if_safe(cache: SharedMemoryCache, cache_key: str, data: Optional[pd.DataFrame]):
  if data is None or data.empty:
    return
  if current_process().name == 'MainProcess':
    cache.put(cache_key, data)


def _slice_market_data_by_time(
    data: pd.DataFrame,
    start_time: Optional[datetime],
    end_time: datetime,
) -> pd.DataFrame:
  time_values = data['time'].values
  end_time_ms = int(end_time.timestamp() * 1000)
  start_idx = 0
  if start_time is not None:
    start_time_ms = int(start_time.timestamp() * 1000)
    start_idx = np.searchsorted(time_values, start_time_ms, side='left')
  end_idx = np.searchsorted(time_values, end_time_ms, side='right')
  return data.iloc[start_idx:end_idx]


def _load_market_window(
    stock_code: str,
    count: int,
    base_time: datetime,
    period: str,
    allow_tainted: bool,
    dividend_type: str,
) -> Optional[pd.DataFrame]:
  data = get_market_data(
    stock_code,
    count,
    base_time,
    period,
    allow_tainted=allow_tainted,
    dividend_type=dividend_type,
  )
  return _filter_active_bars(data)


def _get_cached_market_window(
    stock_code: str,
    count: int,
    base_time: datetime,
    period: str,
    allow_tainted: bool,
    dividend_type: str,
) -> Optional[pd.DataFrame]:
  cache = _get_market_cache(period)
  cache_key = _make_market_cache_key(stock_code, dividend_type)
  cached_data = cache.get(cache_key) if cache.contains(cache_key) else None

  if cached_data is not None and not cached_data.empty:
    cached_slice = _slice_market_data_by_time(cached_data, None, base_time)
    last_bar_time = datetime.fromtimestamp(int(cached_data.iloc[-1]['time']) / 1000)
    if len(cached_slice) >= count and (allow_tainted or is_latest_data(last_bar_time, base_time, period)):
      return cached_data

  fetch_count = count + 2000 if period == '1m' else count
  window_data = _load_market_window(
    stock_code,
    fetch_count,
    base_time,
    period,
    allow_tainted,
    dividend_type,
  )
  if window_data is not None and not window_data.empty:
    _cache_market_data_if_safe(cache, cache_key, window_data)
  return window_data


def init_market_data_range(
    stock_codes: list[str] = None,
    start_time: date | datetime = None,
    end_time: date | datetime = None,
    period: str = '1d',
    max_workers: int = None,
    dividend_type: str = 'back',
):
  """按指定时间窗口预加载共享缓存中的日线数据

  该接口用于回测/GA 这类已知时间范围的场景。
  缓存仍复用现有共享内存键，但只写入截至 ``end_time`` 的最近窗口数据，
  避免为全市场常驻整段上市以来的历史日线。
  """
  if period != '1d':
    raise NotImplementedError("目前仅支持 period='1d'，分钟数据暂不支持")
  if start_time is None or end_time is None:
    raise ValueError('start_time 和 end_time 不能为空')

  start_dt = start_time if isinstance(start_time, datetime) else datetime.combine(start_time, datetime.min.time())
  end_dt = end_time if isinstance(end_time, datetime) else datetime.combine(end_time, datetime.min.time())
  if start_dt > end_dt:
    raise ValueError('start_time 不能晚于 end_time')

  max_workers = max_workers or min(8, os.cpu_count() or 4)
  cache = _get_market_cache(period)
  latest_end_time = get_latest_trading_time(end_dt)
  required_count = len(get_trading_date_span(start_dt.date(), latest_end_time.date()))

  if required_count <= 0:
    core_logger.debug(
      f"跳过窗口预加载：{start_dt.date()} ~ {latest_end_time.date()} 没有交易日"
    )
    return 0

  from .stock_list import allow_buy_stock_code_list
  all_stocks = stock_codes if stock_codes else allow_buy_stock_code_list()
  core_logger.debug(
    f"按窗口预加载 {len(all_stocks)} 只股票的 【{period}/{dividend_type} 数据】到共享内存"
    f"（{start_dt.date()} ~ {latest_end_time.date()}，约 {required_count} 根，{max_workers} 线程）..."
  )

  def _load_one(code):
    cache_key = _make_market_cache_key(code, dividend_type)
    if cache.contains(cache_key):
      return 0
    try:
      data = get_market_data(
        code,
        required_count,
        latest_end_time,
        period,
        allow_tainted=True,
        dividend_type=dividend_type,
      )
    except Exception:
      return 0
    data = _filter_active_bars(data)
    _cache_market_data_if_safe(cache, cache_key, data)
    return 1 if (data is not None and not data.empty) else 0

  loaded = 0
  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(_load_one, code): code for code in all_stocks}
    done = 0
    for future in as_completed(futures):
      loaded += future.result()
      done += 1
      if done % 200 == 0:
        core_logger.debug(f"窗口数据预加载进度: {done}/{len(all_stocks)}")

  # 基准指数也用同一窗口预加载，避免回测阶段为单只指数再落回全量历史读取。
  from utils.stock.info import baseline_stock_code
  baseline_cache_key = _make_market_cache_key(baseline_stock_code, dividend_type)
  if not cache.contains(baseline_cache_key):
    try:
      baseline_data = get_market_data(
        baseline_stock_code,
        required_count,
        latest_end_time,
        period,
        allow_tainted=True,
        dividend_type=dividend_type,
      )
    except Exception:
      baseline_data = None
    baseline_data = _filter_active_bars(baseline_data)
    _cache_market_data_if_safe(cache, baseline_cache_key, baseline_data)

  core_logger.debug(f"窗口预加载完成，新增加载 {loaded} 只。")
  stat = cache.stat()
  core_logger.debug(f"_GLOBAL_DAILY_CACHE 包含 {stat['count']} 条数据，共计 {stat['total_size_mb']:.2f} MB。")
  return loaded


def cleanup_shared_cache():
  """清理共享缓存（在主进程退出时调用）"""
  _GLOBAL_DAILY_CACHE.cleanup()
  _GLOBAL_MINUTE_CACHE.cleanup()


def _is_same_trade_day(bar_time_ms: int, trade_date: datetime | date) -> bool:
  return datetime.fromtimestamp(bar_time_ms / 1000).date() == (
    trade_date.date() if isinstance(trade_date, datetime) else trade_date
  )


def _enforce_strict_trade_date(
    data: Optional[pd.DataFrame],
    trade_date: datetime | date,
) -> Optional[pd.DataFrame]:
  if data is None or data.empty:
    return None

  bar = data.iloc[-1]
  if not _is_same_trade_day(int(bar['time']), trade_date):
    return None
  return data


def _enforce_strict_trade_date_batch(
    data_dict: dict[str, Optional[pd.DataFrame]],
    trade_date: datetime | date,
) -> dict[str, Optional[pd.DataFrame]]:
  return {
    code: _enforce_strict_trade_date(data, trade_date)
    for code, data in data_dict.items()
  }


def get_full_market_data(
    stock_code: str,
    period: str = '1d',
    target_time: datetime = None,
    allow_tainted: bool = True,
    dividend_type: str = 'back'
) -> Optional[pd.DataFrame]:
  """获取股票的完整历史数据（从上市日到目标时间）。

  仅保留给少数确实需要完整历史的调用方；不再写入共享缓存。
  """
  if target_time is None:
    target_time = datetime.now()

  try:
    data = get_market_data(stock_code, None, target_time, period, allow_tainted=allow_tainted, dividend_type=dividend_type)
    return _filter_active_bars(data)
  except Exception:
    return None

def get_market_data_from_cache(
    stock_code: str,
    count: int,
    base_time: datetime,
    period: str = '1d',
    allow_tainted: bool = False,
    dividend_type: str = 'back',
    strict_trade_date: bool = False,
) -> Optional[pd.DataFrame]:
  """从缓存中获取指定时间范围的市场数据

  性能优化：
  - 使用numpy的searchsorted进行二分查找（O(log n)复杂度）
  - 使用iloc切片实现零拷贝视图
  - 避免重复的timestamp转换
  """
  # 历史日线不做阻塞式补数下载，避免大量并发卡死在 xtdata.download_history_data2
  tainted = allow_tainted or (period == '1d' and base_time.date() < date.today())

  # 检查股票日期有效性
  if not check_stock_valid_at_date(stock_code, base_time.date()):
    raise ValueError(f'{stock_code} 获取 {format_qmt_datetime(base_time)} {count}*{period} 失败：股票不存在或在该时间点无效')

  full_data = _get_cached_market_window(
    stock_code,
    count,
    base_time,
    period,
    tainted,
    dividend_type,
  )

  if full_data is None or full_data.empty:
    return None

  # 根据base_time筛选出<=base_time的数据
  # 优化：使用二分查找，因为time列是升序排列的
  base_time_ms = int(base_time.timestamp() * 1000)

  # 使用 searchsorted 进行二分查找，找到第一个 > base_time_ms 的位置
  time_values = full_data['time'].values
  insert_pos = np.searchsorted(time_values, base_time_ms, side='right')

  if insert_pos < count:
    return None

  # 使用 iloc 直接切片，比 tail() 更高效（零拷贝视图）
  start_pos = max(0, insert_pos - count)
  result = full_data.iloc[start_pos:insert_pos]
  if strict_trade_date:
    return _enforce_strict_trade_date(result, base_time)
  return result


def get_market_data_range_from_cache(
    stock_code: str,
    start_time: datetime,
    end_time: datetime,
    period: str = '1d',
    allow_tainted: bool = False,
    dividend_type: str = 'back',
    strict_end_trade_date: bool = False,
    skip_validity_check: bool = False,
) -> Optional[pd.DataFrame]:
  if start_time > end_time:
    raise ValueError('start_time 不能晚于 end_time')
  if period != '1d':
    raise NotImplementedError("目前仅支持 period='1d' 的区间缓存读取")

  if not skip_validity_check and not check_stock_valid_at_date(stock_code, end_time.date()):
    raise ValueError(f'{stock_code} 获取 {format_qmt_datetime(end_time)} 区间数据失败：股票不存在或在该时间点无效')

  tainted = allow_tainted or end_time.date() < date.today()
  required_count = len(get_trading_date_span(start_time.date(), end_time.date()))
  if required_count <= 0:
    return None

  window_data = _get_cached_market_window(
    stock_code,
    required_count,
    end_time,
    period,
    tainted,
    dividend_type,
  )
  if window_data is None or window_data.empty:
    return None

  result = _slice_market_data_by_time(window_data, start_time, end_time)
  if result.empty:
    return None
  if strict_end_trade_date:
    return _enforce_strict_trade_date(result, end_time)
  return result

def get_market_data(
    stock_code: str,
    count: Optional[int],
    base_time: datetime = None,
    period: str = '1d',
    allow_tainted: bool = False,  # 是否允许返回不完整或非最新的数据
    dividend_type: str = 'back',
) -> Optional[pd.DataFrame]:
  """ deprecated, use get_market_data_batch instead """
  # 检查股票日期有效性
  if not check_stock_valid_at_date(stock_code, base_time.date()):
    raise ValueError(f'{stock_code} 获取 {format_qmt_datetime(base_time)} {count}*{period} 失败：股票不存在或在该时间点无效')

  history_data = get_market_data_batch(
    [stock_code], count, base_time, period, allow_tainted, dividend_type
  )[stock_code]
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
    dividend_type: str = 'back',
    strict_trade_date: bool = False,
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

  history_data_dict = get_history_data(stock_codes, count, latest_trading_time, period, dividend_type)

  # 如果允许污染数据，直接返回，跳过所有校验和修复逻辑
  if allow_tainted:
    if strict_trade_date:
      return _enforce_strict_trade_date_batch(history_data_dict, input_time)
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
      updated_dict = get_history_data_after_download(stocks_need_update, count, latest_trading_time, period, dividend_type)

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

      more_data_dict = get_history_data_after_download(stocks_need_suspend_fix, estimated_count * (attempt + 1) + 1, max_earliest_time, period, dividend_type)

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

  if strict_trade_date:
    return _enforce_strict_trade_date_batch(history_data_dict, input_time)
  return history_data_dict
