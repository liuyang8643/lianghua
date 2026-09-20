"""退市股票数据管理（从预下载 parquet 读取，不联网）"""
from datetime import date
from pathlib import Path
from typing import NamedTuple

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class DelistStockInfo(NamedTuple):
  """退市股票信息"""
  name: str  # 股票简称
  list_date: date  # 上市日期
  delist_date: date  # 退市日期
  # The Shanghai source can publish A/B listing dates under the same A-share
  # company code. Keep every source date; runtime validation resolves the one
  # supported by the A-share bar/event history.
  list_date_candidates: tuple[date, ...] = ()


_DELIST_CACHE = None

_SCHEMAS = {
  'SH': ('公司代码', '公司简称', '上市日期', '暂停上市日期'),
  'SZ': ('证券代码', '证券简称', '上市日期', '终止上市日期'),
}


def _parse_delist_frame(df) -> dict[str, DelistStockInfo]:
  import pandas as pd

  if 'exchange' not in df.columns:
    raise ValueError('退市数据缺少 exchange 列')

  result = {}
  for exchange, columns in _SCHEMAS.items():
    code_col, name_col, list_col, delist_col = columns
    rows = df[df['exchange'] == exchange]
    if rows.empty:
      continue
    missing = set(columns) - set(rows.columns)
    if missing:
      raise ValueError(f'退市数据 {exchange} 缺少列: {sorted(missing)}')
    if rows[list(columns)].isna().any().any():
      raise ValueError(f'退市数据 {exchange} 存在空字段')

    codes = rows[code_col].astype(str).str.strip().str.zfill(6)
    if not codes.str.fullmatch(r'\d{6}').all():
      raise ValueError(f'退市数据 {exchange} 存在非法代码')
    names = rows[name_col].astype(str).str.strip()
    if names.eq('').any():
      raise ValueError(f'退市数据 {exchange} 存在空简称')
    list_dates = pd.to_datetime(rows[list_col], format='%Y-%m-%d', errors='raise').dt.date
    delist_dates = pd.to_datetime(rows[delist_col], format='%Y-%m-%d', errors='raise').dt.date
    for code, name, list_date, delist_date in zip(codes, names, list_dates, delist_dates):
      stock_code = f'{code}.{exchange}'
      if stock_code not in result:
        result[stock_code] = DelistStockInfo(
          name, list_date, delist_date, (list_date,),
        )
        continue
      previous = result[stock_code]
      if previous.name != name or previous.delist_date != delist_date:
        raise ValueError(f'退市数据 {stock_code} 存在冲突记录')
      candidates = tuple(sorted({
        previous.list_date,
        *previous.list_date_candidates,
        list_date,
      }))
      result[stock_code] = DelistStockInfo(
        name,
        min(candidates),
        delist_date,
        candidates,
      )

  unknown = set(df['exchange'].dropna().unique()) - set(_SCHEMAS)
  if unknown:
    raise ValueError(f'退市数据存在未知交易所: {sorted(unknown)}')
  if not result:
    raise ValueError('退市数据为空')
  return result

def get_delist_stock_info() -> dict[str, DelistStockInfo]:
  """从预下载 parquet 读取退市股票信息（进程内 lru_cache）。

  Returns:
    {股票代码: DelistStockInfo(name, list_date, delist_date)}
  """
  global _DELIST_CACHE
  if _DELIST_CACHE is not None:
    return _DELIST_CACHE

  import pandas as pd

  path = _DATA_DIR / "delist" / "delist.parquet"
  if not path.exists():
    raise FileNotFoundError(f'退市数据缺失: {path}')

  _DELIST_CACHE = _parse_delist_frame(pd.read_parquet(path))
  return _DELIST_CACHE


def invalidate_delist_cache() -> None:
  """Drop the process-local view after an atomic offline-data replacement."""
  global _DELIST_CACHE
  _DELIST_CACHE = None


__all__ = [
  'DelistStockInfo',
  'get_delist_stock_info',
  'invalidate_delist_cache',
]
