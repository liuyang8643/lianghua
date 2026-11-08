from datetime import datetime, date
from pathlib import Path
import re
from typing import Optional, Union

import pandas as pd

import logging

data_logger = logging.getLogger(__name__)
from utils.stock.info import is_b_stock
from data.db.delist import get_delist_stock_info

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_STOCK_CODE_PATTERN = re.compile(r"^\d{6}\.(SH|SZ|BJ)$")


def validate_stock_list_frame(frame: pd.DataFrame) -> pd.DataFrame:
  """Validate and canonicalize one complete current A-share snapshot."""
  required = {"stock_code", "exchange"}
  missing = required.difference(frame.columns)
  if missing:
    raise ValueError(f"stock_list 缺少列: {sorted(missing)}")
  if frame.empty:
    raise ValueError("stock_list 不得为空")

  result = frame.loc[:, ["stock_code", "exchange"]].copy()
  result["stock_code"] = result["stock_code"].astype(str).str.strip().str.upper()
  result["exchange"] = result["exchange"].astype(str).str.strip().str.upper()
  invalid = ~result["stock_code"].map(lambda value: bool(_STOCK_CODE_PATTERN.fullmatch(value)))
  if invalid.any():
    raise ValueError(
      "stock_list 包含非法股票代码: "
      + ", ".join(result.loc[invalid, "stock_code"].head(20))
    )
  expected_exchange = result["stock_code"].str.rsplit(".", n=1).str[-1]
  if not result["exchange"].equals(expected_exchange):
    raise ValueError("stock_list exchange 与 stock_code 后缀不一致")
  if result["stock_code"].duplicated().any():
    duplicates = result.loc[result["stock_code"].duplicated(), "stock_code"]
    raise ValueError(
      "stock_list 股票代码重复: " + ", ".join(duplicates.head(20))
    )
  return result.sort_values("stock_code", kind="stable").reset_index(drop=True)


def load_current_stock_codes() -> tuple[str, ...]:
  """Read the validated current-market stock axis without adding delisted names."""
  path = _DATA_DIR / "stock_list" / "stock_list.parquet"
  if not path.exists():
    raise FileNotFoundError(f"stock_list 快照不存在: {path}")
  frame = validate_stock_list_frame(pd.read_parquet(path))
  return tuple(frame["stock_code"].tolist())


def _fetch_all_a_stocks() -> tuple[str, ...]:
  """获取全部A股股票代码（从预下载 parquet 读取）"""
  path = _DATA_DIR / "stock_list" / "stock_list.parquet"
  if path.exists():
    codes = list(load_current_stock_codes())
    data_logger.debug(f"A股全部股票: {len(codes)} 只 (parquet)")
    return tuple(codes)
  return ()


def _get_stock_date_range(stock_code: str) -> Optional[tuple[date, Optional[date]]]:
  """获取股票有效日期范围: (上市日期, 退市日期)，退市日期为None表示未退市"""
  # 优先使用 akshare 退市数据，避免对退市股票调用接口
  delist_info = get_delist_stock_info()
  if stock_code in delist_info:
    info = delist_info[stock_code]
    return info.list_date, info.delist_date

  # 非退市股票：当前正常交易中，不必逐只查 xtdata
  return date(1990, 1, 1), None


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
    data_logger.debug(f"获取 {target_date.strftime('%Y-%m-%d')} 有效股票: {len(stocks)} -> {len(filtered)}(排除B股) -> {len(result)}(日期过滤)")
  else:
    result = tuple(sorted(filtered))
    data_logger.debug(f"获取所有股票: {len(stocks)} -> {len(filtered)}(排除B股)")

  return list(result)
