"""
因子计算数据持久化缓存层
使用 Parquet 格式存储 + LRU 内存缓存
"""
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import hashlib
import pickle
import sys
from pandas import DataFrame
import pandas as pd

# 缓存配置
CACHE_DIR = Path(__file__).parent / '.cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# LRU 缓存配置：8GB 内存限制
# 假设平均每个 DataFrame 约 100KB，8GB = 8 * 1024 * 1024 KB = 8388608 KB
# 最大缓存条目数 = 8388608 / 100 ≈ 83886
LRU_MAX_SIZE = 80000  # 留一些余量

class CacheKey:
  """缓存键生成器"""

  @staticmethod
  def make_key(code: str, base_time: datetime, method: str, *args, **kwargs) -> str:
    """
    生成缓存键
    :param code: 股票代码
    :param base_time: 基准时间
    :param method: 方法名
    :param args: 位置参数
    :param kwargs: 关键字参数
    :return: 缓存键字符串
    """
    # 使用日期部分作为key（忽略时分秒）
    date_str = base_time.strftime('%Y%m%d')

    # 序列化参数
    params_str = f"{args}_{sorted(kwargs.items())}"
    param_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]

    return f"{code}_{date_str}_{method}_{param_hash}"

  @staticmethod
  def make_file_path(key: str, data_type: str = 'df') -> Path:
    """
    生成缓存文件路径
    :param key: 缓存键
    :param data_type: 数据类型 ('df' for DataFrame, 'pkl' for pickle)
    :return: 文件路径
    """
    # 按股票代码分目录存储
    code = key.split('_')[0]
    code_dir = CACHE_DIR / code
    code_dir.mkdir(exist_ok=True)

    if data_type == 'df':
      return code_dir / f"{key}.parquet"
    else:
      return code_dir / f"{key}.pkl"

class DiskCache:
  """磁盘缓存管理器"""

  @staticmethod
  def save_dataframe(key: str, df: DataFrame) -> None:
    """保存 DataFrame 到磁盘"""
    file_path = CacheKey.make_file_path(key, 'df')
    try:
      df.to_parquet(file_path, engine='pyarrow', compression='snappy')
    except Exception as e:
      # 静默失败，不影响主流程
      pass

  @staticmethod
  def load_dataframe(key: str) -> Optional[DataFrame]:
    """从磁盘加载 DataFrame"""
    file_path = CacheKey.make_file_path(key, 'df')
    if not file_path.exists():
      return None

    try:
      return pd.read_parquet(file_path, engine='pyarrow')
    except Exception as e:
      # 缓存损坏，删除文件
      file_path.unlink(missing_ok=True)
      return None

  @staticmethod
  def save_pickle(key: str, data: Any) -> None:
    """保存 pickle 数据到磁盘"""
    file_path = CacheKey.make_file_path(key, 'pkl')
    try:
      with open(file_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)  # type: ignore
    except Exception as e:
      pass

  @staticmethod
  def load_pickle(key: str) -> Optional[Any]:
    """从磁盘加载 pickle 数据"""
    file_path = CacheKey.make_file_path(key, 'pkl')
    if not file_path.exists():
      return None

    try:
      with open(file_path, 'rb') as f:
        return pickle.load(f)
    except Exception as e:
      file_path.unlink(missing_ok=True)
      return None

class MemoryCache:
  """内存 LRU 缓存管理器"""

  # 使用类级别的缓存字典，所有实例共享
  _df_cache = {}
  _pkl_cache = {}
  _cache_size = 0

  @classmethod
  def _estimate_size(cls, obj: Any) -> int:
    """估算对象大小（字节）"""
    if isinstance(obj, DataFrame):
      return obj.memory_usage(deep=True).sum()
    else:
      return sys.getsizeof(obj)

  @classmethod
  def _evict_if_needed(cls, new_size: int) -> None:
    """如果需要，驱逐最旧的缓存项"""
    max_bytes = 8 * 1024 * 1024 * 1024  # 8GB

    while cls._cache_size + new_size > max_bytes and (cls._df_cache or cls._pkl_cache):
      # 简单的 FIFO 策略，删除第一个元素
      if cls._df_cache:
        key, (_, size) = next(iter(cls._df_cache.items()))
        del cls._df_cache[key]
        cls._cache_size -= size
      elif cls._pkl_cache:
        key, (_, size) = next(iter(cls._pkl_cache.items()))
        del cls._pkl_cache[key]
        cls._cache_size -= size

  @classmethod
  def get_dataframe(cls, key: str) -> Optional[DataFrame]:
    """从内存缓存获取 DataFrame"""
    if key in cls._df_cache:
      return cls._df_cache[key][0].copy()  # 返回副本避免修改
    return None

  @classmethod
  def set_dataframe(cls, key: str, df: DataFrame) -> None:
    """设置 DataFrame 到内存缓存"""
    size = cls._estimate_size(df)
    cls._evict_if_needed(size)
    cls._df_cache[key] = (df.copy(), size)  # 存储副本
    cls._cache_size += size

  @classmethod
  def get_pickle(cls, key: str) -> Optional[Any]:
    """从内存缓存获取 pickle 数据"""
    if key in cls._pkl_cache:
      return cls._pkl_cache[key][0]
    return None

  @classmethod
  def set_pickle(cls, key: str, data: Any) -> None:
    """设置 pickle 数据到内存缓存"""
    size = cls._estimate_size(data)
    cls._evict_if_needed(size)
    cls._pkl_cache[key] = (data, size)
    cls._cache_size += size

  @classmethod
  def clear(cls) -> None:
    """清空所有缓存"""
    cls._df_cache.clear()
    cls._pkl_cache.clear()
    cls._cache_size = 0

def cached_dataframe(method_name: str):
  """
  DataFrame 缓存装饰器
  用于装饰返回 DataFrame 的方法
  """

  def decorator(func):
    def wrapper(self, *args, **kwargs):
      # 生成缓存键
      cache_key = CacheKey.make_key(
        self.code,
        self.base_time,
        method_name,
        *args,
        **kwargs
      )

      # 1. 尝试从内存缓存读取
      result = MemoryCache.get_dataframe(cache_key)
      if result is not None:
        return result

      # 2. 尝试从磁盘缓存读取
      result = DiskCache.load_dataframe(cache_key)
      if result is not None:
        # 加载到内存缓存
        MemoryCache.set_dataframe(cache_key, result)
        return result

      # 3. 执行原始方法
      result = func(self, *args, **kwargs)

      # 4. 保存到缓存
      if result is not None and not result.empty:
        MemoryCache.set_dataframe(cache_key, result)
        DiskCache.save_dataframe(cache_key, result)

      return result

    return wrapper

  return decorator

def cached_value(method_name: str):
  """
  值缓存装饰器
  用于装饰返回非 DataFrame 的方法（如 float, tuple, list 等）
  """

  def decorator(func):
    def wrapper(self, *args, **kwargs):
      # 生成缓存键
      cache_key = CacheKey.make_key(
        self.code,
        self.base_time,
        method_name,
        *args,
        **kwargs
      )

      # 1. 尝试从内存缓存读取
      result = MemoryCache.get_pickle(cache_key)
      if result is not None:
        return result

      # 2. 尝试从磁盘缓存读取
      result = DiskCache.load_pickle(cache_key)
      if result is not None:
        # 加载到内存缓存
        MemoryCache.set_pickle(cache_key, result)
        return result

      # 3. 执行原始方法
      result = func(self, *args, **kwargs)

      # 4. 保存到缓存
      if result is not None:
        MemoryCache.set_pickle(cache_key, result)
        DiskCache.save_pickle(cache_key, result)

      return result

    return wrapper

  return decorator
