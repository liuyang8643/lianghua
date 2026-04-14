from datetime import datetime, time
from typing import Optional

import numpy as np
from pandas import DataFrame
from xtquant import xtdata

from utils.stock.format import format_qmt_date, format_qmt_datetime
from utils.stock.time import get_latest_trading_time, get_target_period_backward, get_trading_date_span, is_latest_data
from .stock_list import _get_stock_date_range


def get_history_data(
    stock_codes: list[str],
    count: Optional[int],
    base_time: datetime,
    period: str,
    dividend_type: str = 'back',
) -> dict[str, Optional[DataFrame]]:
  # 当 count 为 None 时，使用 -1 表示直接读取本地已有的全部原始数据。
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


def _build_reliable_bar_mask(data: Optional[DataFrame]) -> Optional[np.ndarray]:
  """构造“可返回给上层”的可靠 bar 掩码。

  QMT 的停牌日线返回并不稳定：
  - 有时停牌日直接缺 bar；
  - 有时会返回 OHLC 全 0 的占位 bar；
  - ``suspendFlag`` 也不总是 1，且 -1 表示当日起复牌，不能按停牌过滤。

  因此最终返回给上层的数据不能只依赖 ``suspendFlag == 1``，
  需要同时剔除“明确停牌”和“全零占位 bar”。
  """
  if data is None or data.empty:
    return None

  mask = np.ones(len(data), dtype=bool)

  if 'suspendFlag' in data.columns:
    suspend_flags = data['suspendFlag'].to_numpy(copy=False)
    mask &= (suspend_flags != 1)

  if all(col in data.columns for col in ('open', 'high', 'low', 'close')):
    open_values = data['open'].to_numpy(copy=False)
    high_values = data['high'].to_numpy(copy=False)
    low_values = data['low'].to_numpy(copy=False)
    close_values = data['close'].to_numpy(copy=False)
    zero_placeholder = (
      (open_values <= 0)
      & (high_values <= 0)
      & (low_values <= 0)
      & (close_values <= 0)
    )
    mask &= ~zero_placeholder

  return mask


def _count_reliable_bars(data: Optional[DataFrame]) -> int:
  if data is None or data.empty:
    return 0
  reliable_mask = _build_reliable_bar_mask(data)
  return int(reliable_mask.sum()) if reliable_mask is not None else 0


def _get_history_signature(
    data: Optional[DataFrame],
) -> tuple[int, int, Optional[datetime], Optional[datetime]]:
  if data is None or data.empty:
    return 0, 0, None, None
  return len(data), _count_reliable_bars(data), _get_first_bar_datetime(data), _get_last_bar_datetime(data)


def _get_full_history_start(stock_code: str, period: str) -> Optional[datetime]:
  if period != '1d':
    return None
  date_range = _get_stock_date_range(stock_code)
  if not date_range or date_range[0] is None:
    return None
  return datetime.combine(date_range[0], time.min)


def _get_expected_history_count(
    stock_code: str,
    base_time: datetime,
    period: str,
    count: int,
) -> int:
  full_history_start = _get_full_history_start(stock_code, period)
  if full_history_start is None:
    return count

  latest_trading_date = get_latest_trading_time(base_time).date()
  if full_history_start.date() > latest_trading_date:
    return 0
  available_count = len(get_trading_date_span(full_history_start.date(), latest_trading_date))
  return min(count, available_count)


def _get_count_history_start(
    stock_code: str,
    base_time: datetime,
    period: str,
    count: int,
) -> datetime:
  required_start = get_target_period_backward(base_time, period, count)
  full_history_start = _get_full_history_start(stock_code, period)
  if full_history_start is None:
    return required_start
  return max(required_start, full_history_start)


def _resolve_download_start(
    stock_code: str,
    data: Optional[DataFrame],
    count: Optional[int],
    base_time: datetime,
    period: str,
) -> Optional[datetime]:
  """计算最小必要下载起点。

  这里同时处理两个坑：
  - ``get_market_data_ex(..., count=N)`` 会继续向前取已有 bar，缺失停牌日不一定影响 raw 数量；
  - 但如果返回里混入了全零占位 bar，它们会占掉 count 名额，导致“raw 足够、可靠 bar 不够”。

  因此下载判定必须同时看：
  - 原始尾部是否已经更新到 ``base_time``；
  - 过滤停牌/占位 bar 后的可靠数量是否达到目标。
  """
  candidates: list[datetime] = []

  if count is None:
    required_start = _get_full_history_start(stock_code, period)
    if required_start is None:
      return None
    if data is None or data.empty:
      return required_start

    first_dt = _get_first_bar_datetime(data)
    if first_dt is not None and first_dt.date() > required_start.date():
      candidates.append(required_start)

    last_dt = _get_last_bar_datetime(data)
    if last_dt is not None and not is_latest_data(last_dt, base_time, period):
      candidates.append(last_dt)

    return min(candidates) if candidates else None

  expected_count = _get_expected_history_count(stock_code, base_time, period, count)
  if expected_count <= 0:
    return None

  required_start = _get_count_history_start(stock_code, base_time, period, count)
  if data is None or data.empty:
    return required_start

  first_dt = _get_first_bar_datetime(data)
  if first_dt is None:
    return required_start

  reliable_count = _count_reliable_bars(data)
  if reliable_count < expected_count:
    missing_count = expected_count - reliable_count
    prepend_start = get_target_period_backward(first_dt, period, missing_count)
    full_history_start = _get_full_history_start(stock_code, period)
    if full_history_start is not None:
      prepend_start = max(prepend_start, full_history_start)
    if prepend_start < first_dt:
      candidates.append(prepend_start)

  last_dt = _get_last_bar_datetime(data)
  if last_dt is not None and not is_latest_data(last_dt, base_time, period):
    candidates.append(last_dt)

  return min(candidates) if candidates else None


def _get_next_fetch_count(
    data_dict: dict[str, Optional[DataFrame]],
    stock_codes: list[str],
    requested_count: Optional[int],
    base_time: datetime,
    period: str,
    current_fetch_count: Optional[int],
) -> Optional[int]:
  if requested_count is None or current_fetch_count is None:
    return current_fetch_count

  next_fetch_count = current_fetch_count
  for code in stock_codes:
    data = data_dict.get(code)
    expected_count = _get_expected_history_count(code, base_time, period, requested_count)
    if expected_count <= 0:
      continue

    reliable_count = _count_reliable_bars(data)
    if reliable_count >= expected_count:
      continue

    data_size = 0 if data is None or data.empty else len(data)
    # 当读取结果正好被 count 截断时，优先扩大本地读取窗口；
    # 许多停牌 case 的更早有效 bar 已经在本地，只是被尾部停牌/raw 占位 bar 挤掉了。
    if data_size < current_fetch_count:
      continue
    invalid_count = max(0, data_size - reliable_count)
    next_fetch_count = max(
      next_fetch_count,
      current_fetch_count + (expected_count - reliable_count) + invalid_count,
    )

  return next_fetch_count


def get_history_data_after_download(
    stock_codes: list[str],
    count: Optional[int],
    base_time: datetime,
    period: str,
    dividend_type: str = 'back',
) -> dict[str, Optional[DataFrame]]:
  """按需补齐本地原始行情，再返回原始结果。

  这里故意只在“确实缺数或尾部过期”时才调用 ``download_history_data2``，
  因为下载很慢。下载起点按每只股票单独计算，并按起点分组，避免整批股票重复回补。

  另外，当前 QMT 上 ``download_history_data2`` 的 ``incrementally`` 参数实际无效，
  这里直接忽略，不再传递这个参数。
  """
  fetch_count = count
  existing = get_history_data(stock_codes, fetch_count, base_time, period, dividend_type)

  # 最多允许少量“扩本地读取窗口 / 触发下载”交替迭代，避免异常数据形态下死循环。
  # 实际正常路径通常 1~2 轮就结束；这里留 6 轮只是保险上限，不承载业务语义。
  for _ in range(6):
    next_fetch_count = _get_next_fetch_count(
      existing,
      stock_codes,
      count,
      base_time,
      period,
      fetch_count,
    )
    if next_fetch_count is not None and next_fetch_count > fetch_count:
      fetch_count = next_fetch_count
      existing = get_history_data(stock_codes, fetch_count, base_time, period, dividend_type)
      continue

    stocks_need_download: dict[str, datetime] = {}
    for code in stock_codes:
      start_dt = _resolve_download_start(code, existing.get(code), count, base_time, period)
      if start_dt is not None:
        stocks_need_download[code] = start_dt

    if not stocks_need_download:
      break

    end_time_str = _format_download_time(base_time, period)
    download_groups: dict[str, list[str]] = {}
    for code, start_dt in stocks_need_download.items():
      start_time_str = _format_download_time(start_dt, period)
      download_groups.setdefault(start_time_str, []).append(code)

    previous_signatures = {
      code: _get_history_signature(existing.get(code))
      for code in stocks_need_download
    }

    for start_time_str, grouped_codes in download_groups.items():
      xtdata.download_history_data2(
        grouped_codes,
        period,
        start_time=start_time_str,
        end_time=end_time_str,
      )

    existing = get_history_data(stock_codes, fetch_count, base_time, period, dividend_type)
    if all(_get_history_signature(existing.get(code)) == previous_signatures[code] for code in stocks_need_download):
      break

  return existing
