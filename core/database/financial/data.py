"""
财务数据加载模块
从本地 parquet 读取财务数据，提供单股票查询接口
防止数据泄露：只返回披露日期 <= 查询日期的数据
"""
from datetime import date
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd

_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "financial"

_FIN_MEM_CACHE: dict[str, pd.DataFrame] = {}


def _pershare_path() -> Path:
    return _DATA_DIR / "pershare_index.parquet"


def get_financial_data(stock_code: str, table_name: str) -> Optional[pd.DataFrame]:
    """获取指定股票的财务数据（从本地 parquet 读取）。

    Args:
        stock_code: 股票代码（如 '600051.SH'）
        table_name: 表名（仅支持 PershareIndex，Income/Balance/CashFlow 已废弃）

    Returns:
        DataFrame: 财务数据
    """
    if table_name != "PershareIndex":
        return None

    path = _pershare_path()
    if not path.exists():
        return None

    if stock_code not in _FIN_MEM_CACHE:
        df_all = pd.read_parquet(path)
        for code, group in df_all.groupby("stock_code"):
            _FIN_MEM_CACHE[code] = group.reset_index(drop=True)
        # 为不存在的股票打标记
        if stock_code not in _FIN_MEM_CACHE:
            _FIN_MEM_CACHE[stock_code] = pd.DataFrame()

    df = _FIN_MEM_CACHE.get(stock_code)
    if df is None or df.empty:
        return None
    return df


def get_financial_indicator(
    stock_code: str,
    query_date: date,
    indicator_name: str,
    table_name: str = "PershareIndex",
    use_announce_date: bool = True
) -> Optional[float]:
    """获取单个财务指标，只返回披露日期 <= 查询日期的数据。"""
    df = get_financial_data(stock_code, table_name)
    if df is None or df.empty:
        return None

    date_col = 'm_anntime' if use_announce_date else 'm_timetag'
    valid_data = df[df[date_col] <= pd.Timestamp(query_date)]
    if valid_data.empty:
        return None

    latest_record = valid_data.iloc[-1]
    if indicator_name not in latest_record.index:
        return None

    value = latest_record[indicator_name]
    return None if pd.isna(value) else float(value)


def get_financial_indicators(
    stock_code: str,
    query_date: date,
    indicator_names: List[str],
    table_name: str = "PershareIndex",
    use_announce_date: bool = True
) -> Dict[str, Optional[float]]:
    """批量获取多个财务指标。"""
    df = get_financial_data(stock_code, table_name)
    if df is None or df.empty:
        return {name: None for name in indicator_names}

    date_col = 'm_anntime' if use_announce_date else 'm_timetag'
    valid_data = df[df[date_col] <= pd.Timestamp(query_date)]
    if valid_data.empty:
        return {name: None for name in indicator_names}

    latest_record = valid_data.iloc[-1]
    result = {}
    for name in indicator_names:
        if name not in latest_record.index:
            result[name] = None
        else:
            value = latest_record[name]
            result[name] = None if pd.isna(value) else float(value)
    return result


# ==================== 便捷函数 ====================

def get_roe(stock_code: str, query_date: date, use_announce_date: bool = True) -> Optional[float]:
    return get_financial_indicator(stock_code, query_date, 'du_return_on_equity', use_announce_date=use_announce_date)

def get_eps(stock_code: str, query_date: date, use_announce_date: bool = True) -> Optional[float]:
    return get_financial_indicator(stock_code, query_date, 's_fa_eps_basic', use_announce_date=use_announce_date)
