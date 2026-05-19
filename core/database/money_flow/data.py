"""
资金流向数据加载模块
从本地 parquet 读取资金流向数据
"""
from datetime import datetime, date
from pathlib import Path
from typing import Optional
from functools import lru_cache

import pandas as pd

_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "money_flow" / "money_flow.parquet"

# 列类型映射
_COLUMN_DTYPES = {
    'code': str,
    'name': str,
    'date': str,
    '主动买入特大单金额（元）': float,
    '被动买入特大单金额（元）': float,
    '主动买入大单金额（元）': float,
    '被动买入大单金额（元）': float,
    '主动买入中单金额（元）': float,
    '被动买入中单金额（元）': float,
    '主动卖出特大单金额（元）': float,
    '被动卖出特大单金额（元）': float,
    '主动卖出大单金额（元）': float,
    '被动卖出大单金额（元）': float,
    '主动卖出中单金额（元）': float,
    '被动卖出中单金额（元）': float,
    '小单买入金额（元）': float,
    '小单卖出金额（元）': float,
    'DDE大单净额（元）': float,
    '金额流入率（%）': float,
    '大单净量（流通股%）': float,
}

_full_df: pd.DataFrame | None = None
_date_cache: dict[str, pd.DataFrame] = {}


def _load_full_df() -> pd.DataFrame | None:
    global _full_df
    if _full_df is not None:
        return _full_df
    if not _DATA_PATH.exists():
        return None
    _full_df = pd.read_parquet(_DATA_PATH)
    _full_df['code'] = _full_df['code'].astype(str).str.replace('.0', '', regex=False).str.zfill(6)
    return _full_df


@lru_cache(maxsize=256)
def get_money_flow_data(target_date: date | datetime) -> Optional[pd.DataFrame]:
    """获取指定日期的资金流向数据（从本地 parquet 读取）。"""
    if isinstance(target_date, datetime):
        target_date = target_date.date()

    date_str = target_date.isoformat()
    if date_str in _date_cache:
        return _date_cache[date_str]

    df_all = _load_full_df()
    if df_all is None:
        return None

    df = df_all[df_all['date'] == date_str]
    if df.empty:
        return None

    _date_cache[date_str] = df
    return df
