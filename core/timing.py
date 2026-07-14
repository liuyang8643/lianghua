"""大盘择时调仓，共享给回测与实盘。"""

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

INDEX_INFO = {
  'sh000001': '上证指数',
  'H00300': '沪深300全收益',
  'sh000905': '中证500',
  'sh000852': '中证1000',
}


def _index_parquet_path(symbol):
  return DATA_DIR / f"index_{symbol}_daily.parquet"


def _read_index_parquet(symbol, price_col='open'):
  import pyarrow.parquet as pq
  path = _index_parquet_path(symbol)
  if not path.exists():
    raise FileNotFoundError(f"指数数据缺失: {path}，请先运行 data/update_all.py 预下载")
  table = pq.read_table(path)
  dates_arr = table.column('trade_date').to_numpy().astype('datetime64[D]')
  if price_col not in table.column_names:
    price_col = 'close'
  price_arr = table.column(price_col).to_numpy().astype(np.float64)
  return dates_arr, price_arr


def load_index_open(symbol, trade_dates=None):
  """加载指数开盘价（从预下载 parquet，不联网）。返回 (dates_array, open_array)。"""
  dates_arr, open_arr = _read_index_parquet(symbol, 'open')

  if trade_dates is None:
    return dates_arr, open_arr

  date_to_val = {}
  for i in range(len(dates_arr)):
    date_to_val[dates_arr[i].item()] = open_arr[i]

  missing = [d.date() for d in trade_dates if d.date() not in date_to_val]
  if missing:
    raise ValueError(f"指数 {symbol} 缺少交易日 open: {missing[:5]}")

  aligned = np.array([date_to_val[d.date()] for d in trade_dates], dtype=np.float64)
  return np.array(trade_dates), aligned


def compute_position_multiplier(index_open, window=20, base=0.5, leverage=10,
                                direction=1, floor=0.0, cap=1.0):
  """仓位乘数：窗口收益率 × 杠杆 + 基准仓位，clip 到 [floor, cap]。"""
  arr = np.asarray(index_open, dtype=np.float64)
  n = len(arr)
  ret = np.full(n, np.nan)

  for i in range(window, n):
    ret[i] = (arr[i] - arr[i - window]) / arr[i - window]

  multiplier = np.full(n, base)
  valid = ~np.isnan(ret)
  multiplier[valid] = base + direction * ret[valid] * leverage
  multiplier = np.clip(multiplier, floor, cap)
  return multiplier


def compute_calendar_empty_mask(trade_dates, empty_months) -> np.ndarray | None:
  """日历空仓掩码：指定月份返回 0.0，其余返回 1.0。None 表示不启用。"""
  if not empty_months:
    return None
  months_set = set(empty_months)
  mask = np.ones(len(trade_dates), dtype=np.float64)
  for i, d in enumerate(trade_dates):
    if d.month in months_set:
      mask[i] = 0.0
  return mask


def compute_position_multiplier_for_date(config: dict, target_date) -> float:
  empty_months = config.get('empty_months')
  if empty_months and target_date.month in empty_months:
    return 0.0

  cash_reserve = float(config.get('cash_reserve_ratio', 0.0))
  invest_ratio = 1.0 - cash_reserve

  timing_enabled = config['timing_enabled'] if 'timing_enabled' in config else False
  timing_base = config['timing_base'] if 'timing_base' in config else None
  if not (timing_enabled and timing_base is not None):
    return invest_ratio

  symbol = config['timing_index'] if 'timing_index' in config else 'sh000001'
  dates_arr, open_arr = load_index_open(symbol)
  py_dates = [d.item() for d in dates_arr]
  row = py_dates.index(target_date)
  mult = compute_position_multiplier(
    open_arr,
    window=config['timing_window'] if 'timing_window' in config else 20,
    base=timing_base,
    leverage=config['timing_leverage'] if 'timing_leverage' in config else 10,
    direction=config['timing_direction'] if 'timing_direction' in config else 1,
  )
  return float(mult[row]) * invest_ratio
