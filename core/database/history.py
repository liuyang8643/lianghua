from datetime import datetime, time
from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from utils.stock.time import get_latest_trading_time, get_target_period_backward, get_trading_date_span, is_latest_data
from .stock_list import _get_stock_date_range


def _to_mootdx_code(stock_code: str) -> str:
  """转换 WBR 代码格式为 mootdx 格式

  '600000.SH' -> '600000'
  '000001.SZ' -> '000001'
  """
  return stock_code.split('.')[0]


def _mootdx_frequency(period: str) -> int:
  """转换 period 为 mootdx frequency 代码

  9 = 日线, 8 = 1分钟
  """
  if period == '1d':
    return 9
  elif period == '1m':
    return 8
  raise ValueError(f'不支持的周期: {period}')


def _convert_to_wbr(df: pd.DataFrame) -> pd.DataFrame:
  """将 mootdx 返回的 DataFrame 转换为 WBR 标准格式

  mootdx 返回列: open, close, high, low, vol, amount, datetime(index)
  WBR 格式: time(ms int), open, high, low, close, volume, amount
  """
  if df is None or df.empty:
    return pd.DataFrame(columns=_EMPTY_COLS)

  result = pd.DataFrame()
  # mootdx datetime index 是 datetime 对象，转换为毫秒时间戳
  result['time'] = df.index.astype(np.int64) // 10**6
  result['open'] = df['open'].values
  result['high'] = df['high'].values
  result['low'] = df['low'].values
  result['close'] = df['close'].values
  result['volume'] = df['vol'].astype('float64').values
  result['amount'] = df['amount'].values
  result['preClose'] = df['close'].shift(1).fillna(df['close']).values
  return result


def _filter_and_clip_by_time(data: pd.DataFrame, base_time: datetime, count: Optional[int]) -> pd.DataFrame:
  """按 base_time 截断数据并保留尾部 count 条"""
  if data.empty:
    return data
  base_ms = pd.Timestamp(base_time).value // 10**6
  filtered = data[data['time'] <= base_ms]
  if count is not None and len(filtered) > count:
    return filtered.iloc[-count:]
  return filtered


_EMPTY_COLS = ['time', 'open', 'high', 'low', 'close', 'volume', 'amount', 'preClose']


def get_history_data(
    stock_codes: list[str],
    count: Optional[int],
    base_time: datetime,
    period: str,
    dividend_type: str = 'back',
) -> dict[str, Optional[DataFrame]]:
  """通过 mootdx 获取 K-line 数据

  使用通达信本地数据（mootdx），替代 QMT xtdata.get_market_data_ex()。

  Args:
    stock_codes: 股票代码列表
    count: 需要的 bar 数量，None 表示全部
    base_time: 基准时间，数据裁剪到 <= base_time
    period: 周期 ('1d' 或 '1m')
    dividend_type: 复权类型（mootdx 暂不支持复权，保留参数兼容）

  Returns:
    dict[str, Optional[DataFrame]]: 股票代码到 DataFrame 的映射
  """
  from mootdx.quotes import Quotes

  market = Quotes.factory()

  frequency = _mootdx_frequency(period)
  MAX_OFFSET = 800
  result: dict[str, Optional[DataFrame]] = {}

  for code in stock_codes:
    bare_code = _to_mootdx_code(code)
    try:
      if count is None:
        all_bars = []
        offset = 0
        while True:
          bars = market.bars(symbol=bare_code, frequency=frequency, start=offset, offset=MAX_OFFSET)
          if bars is None or bars.empty:
            break
          all_bars.append(bars)
          if len(bars) < MAX_OFFSET:
            break
          offset += MAX_OFFSET

        if not all_bars:
          result[code] = pd.DataFrame(columns=_EMPTY_COLS)
        else:
          combined = pd.concat(all_bars)
          combined = combined[~combined.index.duplicated(keep='first')]
          wbr_df = _convert_to_wbr(combined)
          result[code] = _filter_and_clip_by_time(wbr_df, base_time, count)
      else:
        # mootdx 从最新 bar 往回取，需估算覆盖 base_time 所需的最小 bar 数
        days_back = (datetime.now() - base_time).days if base_time else 0
        fetch_count = max(count, int(days_back * 0.72) + count + 30)
        all_bars = []
        offset = 0
        while len(pd.concat(all_bars) if all_bars else pd.DataFrame()) < fetch_count:
          bars = market.bars(symbol=bare_code, frequency=frequency, start=offset, offset=MAX_OFFSET)
          if bars is None or bars.empty:
            break
          all_bars.append(bars)
          if len(bars) < MAX_OFFSET:
            break
          offset += MAX_OFFSET

        if not all_bars:
          result[code] = pd.DataFrame(columns=_EMPTY_COLS)
        else:
          combined = pd.concat(all_bars)
          combined = combined[~combined.index.duplicated(keep='first')]
          wbr_df = _convert_to_wbr(combined)
          result[code] = _filter_and_clip_by_time(wbr_df, base_time, count)
    except Exception as e:
      result[code] = pd.DataFrame(columns=_EMPTY_COLS)

  return result


def _build_reliable_bar_mask(data: Optional[DataFrame]) -> Optional[np.ndarray]:
  """构造"可返回给上层"的可靠 bar 掩码。

  剔除全零 OHLC 的占位 bar（停牌日可能返回全零 bar）。
  """
  if data is None or data.empty:
    return None

  mask = np.ones(len(data), dtype=bool)

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


def get_history_data_after_download(
    stock_codes: list[str],
    count: Optional[int],
    base_time: datetime,
    period: str,
    dividend_type: str = 'back',
) -> dict[str, Optional[DataFrame]]:
  """获取历史数据（mootdx 版本，无需下载步骤）

  mootdx 直接读取本地通达信数据，无需下载。因此该函数简化为直接调用 get_history_data()。
  """
  return get_history_data(
    stock_codes,
    count,
    base_time,
    period,
    dividend_type,
  )
