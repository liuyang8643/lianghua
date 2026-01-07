"""
财务数据加载模块
从 S3 读取按股票代码拆分的财务数据，并在本地缓存
防止数据泄露：只返回披露日期 <= 查询日期的数据
"""
from datetime import date
from pathlib import Path
from typing import Optional, Dict, List
from functools import lru_cache
import pandas as pd
import boto3

try:
  from configs.env import S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY
except ImportError:
  raise ValueError("S3 配置不存在，请检查 configs/env.py")

# 本地缓存目录
_cache_dir = Path(__file__).parent / ".cache"
_cache_dir.mkdir(parents=True, exist_ok=True)

# 初始化 S3 客户端
_s3_client = boto3.client(
  's3',
  endpoint_url=S3_ENDPOINT,
  aws_access_key_id=S3_ACCESS_KEY,
  aws_secret_access_key=S3_SECRET_KEY,
  region_name='auto'
)

# S3 桶名称
_S3_BUCKET = 'wbr-financial'

# 表名到文件名的映射
_TABLE_MAPPING = {
  "PershareIndex": "pershare_index.csv",
  "Income": "income.csv",
  "Balance": "balance.csv",
  "CashFlow": "cache_flow.csv",
}

@lru_cache(maxsize=512)
def get_financial_data(stock_code: str, table_name: str) -> Optional[pd.DataFrame]:
  """
  获取指定股票的财务数据（优先本地缓存，否则从 S3 下载）

  Args:
      stock_code: 股票代码（如 '600051.SH'）
      table_name: 表名（PershareIndex, Income, Balance, CashFlow）

  Returns:
      DataFrame: 财务数据，包含列：m_timetag, m_anntime, stock_code, 以及其他财务指标
  """
  if table_name not in _TABLE_MAPPING:
    raise ValueError(f"无效的表名: {table_name}，支持: {list(_TABLE_MAPPING.keys())}")

  filename = _TABLE_MAPPING[table_name]

  # 构建缓存路径
  cache_path = _cache_dir / stock_code / filename

  # 检查本地缓存
  if cache_path.exists():
    df = pd.read_csv(cache_path, encoding='utf-8-sig')
    df['m_timetag'] = pd.to_datetime(df['m_timetag'], format='%Y%m%d')
    df['m_anntime'] = pd.to_datetime(df['m_anntime'], format='%Y%m%d')
    return df

  # 从 S3 下载
  cache_path.parent.mkdir(parents=True, exist_ok=True)
  s3_key = f"{stock_code}/{filename}"

  try:
    _s3_client.download_file(_S3_BUCKET, s3_key, str(cache_path))
    df = pd.read_csv(cache_path, encoding='utf-8-sig')
    df['m_timetag'] = pd.to_datetime(df['m_timetag'], format='%Y%m%d')
    df['m_anntime'] = pd.to_datetime(df['m_anntime'], format='%Y%m%d')
    return df
  except Exception as e:
    # 如果文件不存在或下载失败，返回 None
    if cache_path.exists():
      cache_path.unlink()  # 删除可能损坏的缓存文件
    return None

def get_financial_indicator(
    stock_code: str,
    query_date: date,
    indicator_name: str,
    table_name: str = "PershareIndex",
    use_announce_date: bool = True
) -> Optional[float]:
  """
  获取单个财务指标

  防止数据泄露策略：
  - use_announce_date=True: 使用披露日期（m_anntime），只返回披露日期 <= 查询日期的数据
  - use_announce_date=False: 使用报告期（m_timetag），只返回报告期 <= 查询日期的数据

  Args:
      stock_code: 股票代码（如 '600051.SH'）
      query_date: 查询日期
      indicator_name: 指标名称（如 's_fa_eps_basic', 'du_return_on_equity'）
      table_name: 表名（PershareIndex, Income, Balance, CashFlow）
      use_announce_date: 是否使用披露日期（推荐True，更符合实际）

  Returns:
      float: 指标值，如果无数据则返回 None
  """
  df = get_financial_data(stock_code, table_name)
  if df is None or df.empty:
    return None

  # 根据时间策略筛选
  date_col = 'm_anntime' if use_announce_date else 'm_timetag'

  # 筛选出查询日期之前（或等于）的数据
  valid_data = df[df[date_col] <= pd.Timestamp(query_date)]

  if valid_data.empty:
    return None

  # 获取最新的一条记录（披露日期最晚的）
  latest_record = valid_data.iloc[-1]

  # 获取指标值
  if indicator_name not in latest_record.index:
    return None

  value = latest_record[indicator_name]

  # 处理 NaN 值
  if pd.isna(value):
    return None

  return float(value)

def get_financial_indicators(
    stock_code: str,
    query_date: date,
    indicator_names: List[str],
    table_name: str = "PershareIndex",
    use_announce_date: bool = True
) -> Dict[str, Optional[float]]:
  """
  批量获取多个财务指标

  Args:
      stock_code: 股票代码
      query_date: 查询日期
      indicator_names: 指标名称列表
      table_name: 表名
      use_announce_date: 是否使用披露日期

  Returns:
      Dict: {指标名: 指标值（float 或 None）}
  """
  df = get_financial_data(stock_code, table_name)
  if df is None or df.empty:
    return {name: None for name in indicator_names}

  # 根据时间策略筛选
  date_col = 'm_anntime' if use_announce_date else 'm_timetag'

  # 筛选出查询日期之前（或等于）的数据
  valid_data = df[df[date_col] <= pd.Timestamp(query_date)]

  if valid_data.empty:
    return {name: None for name in indicator_names}

  # 获取最新的一条记录
  latest_record = valid_data.iloc[-1]

  # 提取所有指标
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
  """获取ROE（净资产收益率）"""
  return get_financial_indicator(stock_code, query_date, 'du_return_on_equity', 'PershareIndex', use_announce_date)

def get_eps(stock_code: str, query_date: date, use_announce_date: bool = True) -> Optional[float]:
  """获取EPS（每股收益）"""
  return get_financial_indicator(stock_code, query_date, 's_fa_eps_basic', 'PershareIndex', use_announce_date)

def get_bps(stock_code: str, query_date: date, use_announce_date: bool = True) -> Optional[float]:
  """获取BPS（每股净资产）"""
  return get_financial_indicator(stock_code, query_date, 's_fa_bps', 'PershareIndex', use_announce_date)

def get_profit_growth(stock_code: str, query_date: date, use_announce_date: bool = True) -> Optional[float]:
  """获取净利润增长率"""
  return get_financial_indicator(stock_code, query_date, 'du_profit_rate', 'PershareIndex', use_announce_date)

def get_revenue_growth(stock_code: str, query_date: date, use_announce_date: bool = True) -> Optional[float]:
  """获取营收增长率"""
  return get_financial_indicator(stock_code, query_date, 'inc_revenue_rate', 'PershareIndex', use_announce_date)

def get_current_ratio(stock_code: str, query_date: date, use_announce_date: bool = True) -> Optional[float]:
  """获取流动比率"""
  return get_financial_indicator(stock_code, query_date, 'current_ratio', 'Balance', use_announce_date)

def get_gear_ratio(stock_code: str, query_date: date, use_announce_date: bool = True) -> Optional[float]:
  """获取资产负债率"""
  return get_financial_indicator(stock_code, query_date, 'gear_ratio', 'PershareIndex', use_announce_date)

def get_cash_flow_ps(stock_code: str, query_date: date, use_announce_date: bool = True) -> Optional[float]:
  """获取每股现金流"""
  return get_financial_indicator(stock_code, query_date, 's_fa_ocfps', 'PershareIndex', use_announce_date)
