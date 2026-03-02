"""
资金流向数据加载模块
从 S3 读取 CSV 格式的资金流向数据，并在本地缓存
"""
from datetime import datetime, date
from pathlib import Path
from typing import Optional
import pandas as pd
from functools import lru_cache
import boto3
from filelock import FileLock

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

# CSV 列名 → 类型映射，确保 read_csv 后类型正确
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

def _read_money_flow_csv(path: Path) -> pd.DataFrame:
  """读取资金流向 CSV 并确保列类型正确"""
  df = pd.read_csv(path, encoding='utf-8-sig', dtype=_COLUMN_DTYPES)
  # code 列：去掉可能的 '.0' 后缀并补齐 6 位
  df['code'] = df['code'].str.replace('.0', '', regex=False).str.zfill(6)
  return df

@lru_cache(maxsize=256)
def get_money_flow_data(target_date: date | datetime) -> Optional[pd.DataFrame]:
  """获取指定日期的资金流向数据（优先本地缓存，否则从 S3 下载）"""
  # 统一转换为 date 对象
  if isinstance(target_date, datetime):
    target_date = target_date.date()

  year = target_date.year
  month = target_date.month
  date_str = target_date.strftime('%Y-%m-%d')

  # 构建缓存路径
  cache_path = _cache_dir / str(year) / f"{month:02d}" / f"{date_str}.csv"
  lock_path = cache_path.with_suffix('.csv.lock')

  # 使用文件锁保证多进程安全
  with FileLock(str(lock_path), timeout=30):
    # 检查本地缓存
    if cache_path.exists():
      return _read_money_flow_csv(cache_path)

    # 从 S3 下载
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    s3_key = f"{year}/{month:02d}/{date_str}.csv"

    _s3_client.download_file('wbr-money-flow', s3_key, str(cache_path))
    return _read_money_flow_csv(cache_path)

def get_retail_flow_amount(stock_code: str, target_date: date | datetime) -> Optional[float]:
  """
  获取散户资金流向金额（小单买入 + 小单卖出）

  Args:
      stock_code: 股票代码（支持 '000001' 或 '000001.SZ' 格式）
      target_date: 目标日期

  Returns:
      散户资金金额（元），或 None
  """
  df = get_money_flow_data(target_date)
  if df is None:
    return None

  # 处理股票代码格式（去掉 .SZ/.SH 后缀）
  code = stock_code.split('.')[0].zfill(6)

  row = df[df['code'] == code]
  if row.empty:
    return None

  try:
    data = row.iloc[0]
    return data['小单买入金额（元）'] + data['小单卖出金额（元）']
  except (KeyError, ValueError, TypeError):
    return None

def get_retail_net_flow(stock_code: str, target_date: date | datetime) -> Optional[float]:
  """
  获取散户资金净流入（小单买入 - 小单卖出）

  Args:
      stock_code: 股票代码（支持 '000001' 或 '000001.SZ' 格式）
      target_date: 目标日期

  Returns:
      散户净流入金额（元），正值=散户净买入，负值=散户净卖出，或 None
  """
  df = get_money_flow_data(target_date)
  if df is None:
    return None

  code = stock_code.split('.')[0].zfill(6)

  row = df[df['code'] == code]
  if row.empty:
    return None

  try:
    data = row.iloc[0]
    return data['小单买入金额（元）'] - data['小单卖出金额（元）']
  except (KeyError, ValueError, TypeError):
    return None

def get_main_fund_net_inflow(stock_code: str, target_date: date | datetime) -> Optional[float]:
  """
  获取主力资金净流入（[超大单 + 大单]的买入 - 卖出）

  Args:
      stock_code: 股票代码
      target_date: 目标日期

  Returns:
      主力资金净流入（元），或 None
  """
  df = get_money_flow_data(target_date)
  if df is None:
    return None

  # 处理股票代码格式
  code = stock_code.split('.')[0].zfill(6)

  row = df[df['code'] == code]
  if row.empty:
    return None

  try:
    data = row.iloc[0]
    # 主力资金 = 超大单 + 大单
    buy = (data['主动买入特大单金额（元）'] + data['被动买入特大单金额（元）']
           + data['主动买入大单金额（元）'] + data['被动买入大单金额（元）'])
    sell = (data['主动卖出特大单金额（元）'] + data['被动卖出特大单金额（元）']
            + data['主动卖出大单金额（元）'] + data['被动卖出大单金额（元）'])
    return buy - sell
  except (KeyError, ValueError, TypeError):
    return None
