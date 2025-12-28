"""
资金流向数据加载模块
读取外部 CSV 格式的资金流向数据
"""
from datetime import datetime, date
from pathlib import Path
from typing import Optional
import pandas as pd
from functools import lru_cache

# 默认资金流向数据目录
DEFAULT_MONEY_FLOW_DIR = Path(r"C:\Users\doctorl\Downloads\money-flow")

# 全局配置
_money_flow_dir: Path = DEFAULT_MONEY_FLOW_DIR


def set_money_flow_dir(path: str | Path):
    """设置资金流向数据目录"""
    global _money_flow_dir
    _money_flow_dir = Path(path)


def get_money_flow_dir() -> Path:
    """获取资金流向数据目录"""
    return _money_flow_dir


@lru_cache(maxsize=256)
def _load_money_flow_csv(file_path: str) -> Optional[pd.DataFrame]:
    """加载资金流向 CSV 文件（带缓存）"""
    path = Path(file_path)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
        # 将 code 列转为字符串并补零（确保 6 位）
        df['code'] = df['code'].astype(str).str.zfill(6)
        return df
    except Exception:
        return None


def get_money_flow_file_path(target_date: date | datetime) -> Path:
    """获取指定日期的资金流向文件路径"""
    if isinstance(target_date, datetime):
        target_date = target_date.date()

    year = target_date.year
    month = target_date.month
    date_str = target_date.strftime('%Y-%m-%d')

    return _money_flow_dir / str(year) / f"{month:02d}" / f"{date_str}.csv"


def get_money_flow_data(target_date: date | datetime) -> Optional[pd.DataFrame]:
    """获取指定日期的资金流向数据"""
    file_path = get_money_flow_file_path(target_date)
    return _load_money_flow_csv(str(file_path))


def get_stock_money_flow(stock_code: str, target_date: date | datetime) -> Optional[pd.Series]:
    """
    获取单只股票指定日期的资金流向数据

    Args:
        stock_code: 股票代码（支持 '000001' 或 '000001.SZ' 格式）
        target_date: 目标日期

    Returns:
        该股票的资金流向数据行，或 None
    """
    df = get_money_flow_data(target_date)
    if df is None:
        return None

    # 处理股票代码格式（去掉 .SZ/.SH 后缀）
    code = stock_code.split('.')[0].zfill(6)

    row = df[df['code'] == code]
    if row.empty:
        return None

    return row.iloc[0]


def get_retail_flow_amount(stock_code: str, target_date: date | datetime) -> Optional[float]:
    """
    获取散户资金流向金额（小单买入 + 小单卖出）

    Args:
        stock_code: 股票代码
        target_date: 目标日期

    Returns:
        散户资金金额（元），或 None
    """
    row = get_stock_money_flow(stock_code, target_date)
    if row is None:
        return None

    try:
        small_buy = float(row['小单买入金额（元）'])
        small_sell = float(row['小单卖出金额（元）'])
        return small_buy + small_sell
    except (KeyError, ValueError, TypeError):
        return None


def clear_money_flow_cache():
    """清除资金流向数据缓存"""
    _load_money_flow_csv.cache_clear()
