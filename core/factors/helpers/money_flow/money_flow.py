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

try:
  from configs.env import S3_ENDPOINT, S3_BUCKET, S3_ACCESS_KEY, S3_SECRET_KEY
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

  # 检查本地缓存
  if cache_path.exists():
    df = pd.read_csv(cache_path, encoding='utf-8-sig')
    df['code'] = df['code'].astype(str).str.replace('.0', '', regex=False).str.zfill(6)
    return df

  # 从 S3 下载
  cache_path.parent.mkdir(parents=True, exist_ok=True)
  s3_key = f"{year}/{month:02d}/{date_str}.csv"

  _s3_client.download_file(S3_BUCKET, s3_key, str(cache_path))
  df = pd.read_csv(cache_path, encoding='utf-8-sig')
  df['code'] = df['code'].astype(str).str.replace('.0', '', regex=False).str.zfill(6)
  return df

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
    return float(data['小单买入金额（元）']) + float(data['小单卖出金额（元）'])
  except (KeyError, ValueError, TypeError):
    return None
