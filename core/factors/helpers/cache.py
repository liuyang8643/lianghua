"""
因子计算数据持久化缓存层
使用 Parquet 格式存储 + 内存映射（持久化，~1ms）

优化特性：
- 内存映射读取（操作系统页面缓存）
- 原子写入防止并发写入损坏
- 文件存在性缓存减少文件系统调用
"""
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import hashlib
import pickle
import os
import tempfile
import mmap
from pandas import DataFrame
from threading import RLock
from functools import lru_cache
import io
import pyarrow.parquet as pq
import pyarrow as pa
from utils.hash import hash_function_code

# 缓存配置
CACHE_DIR = Path(__file__).parent / '.cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class CacheKey:
  """缓存键生成器"""

  @staticmethod
  def make_key(code: str, base_time: datetime, method: str, *args, **kwargs) -> str:
    """生成缓存键"""
    date_str = base_time.strftime('%Y%m%d')
    if args or kwargs:
      params_str = f"{args}_{sorted(kwargs.items())}"
      param_hash = hashlib.blake2b(params_str.encode(), digest_size=4).hexdigest()
      return f"{code}_{date_str}_{method}_{param_hash}"
    return f"{code}_{date_str}_{method}"

  @staticmethod
  @lru_cache(maxsize=2048)
  def make_file_path(key: str, data_type: str = 'df') -> Path:
    """生成缓存文件路径（带路径缓存）"""
    code = key.split('_')[0]
    code_dir = CACHE_DIR / code
    code_dir.mkdir(exist_ok=True)
    ext = 'parquet' if data_type == 'df' else 'pkl'
    return code_dir / f"{key}.{ext}"

class DiskCache:
  """磁盘缓存管理器（使用内存映射，操作系统页面缓存实现跨进程共享）"""

  # 文件存在性缓存（减少文件系统调用）
  _existence_cache: dict[str, bool] = {}
  _existence_lock = RLock()

  @classmethod
  def _mark_exists(cls, key: str, data_type: str, exists: bool):
    with cls._existence_lock:
      cls._existence_cache[f"{key}_{data_type}"] = exists

  @classmethod
  def _check_exists(cls, key: str, data_type: str) -> Optional[bool]:
    with cls._existence_lock:
      return cls._existence_cache.get(f"{key}_{data_type}")

  @staticmethod
  def save_dataframe(key: str, df: DataFrame) -> None:
    """保存 DataFrame 到磁盘（原子写入）"""
    file_path = CacheKey.make_file_path(key, 'df')
    try:
      # 使用 PyArrow 直接序列化（更快）
      table = pa.Table.from_pandas(df)
      buffer = io.BytesIO()
      pq.write_table(table, buffer, compression='snappy')
      data = buffer.getvalue()

      # 使用临时文件 + 重命名实现原子写入
      fd, tmp_path = tempfile.mkstemp(suffix='.parquet', dir=file_path.parent)
      try:
        os.write(fd, data)
        os.close(fd)
        if file_path.exists():
          try:
            file_path.unlink()
          except Exception:
            pass
        os.replace(tmp_path, file_path)
        DiskCache._mark_exists(key, 'df', True)
      except Exception:
        try:
          os.close(fd)
        except Exception:
          pass
        try:
          os.unlink(tmp_path)
        except Exception:
          pass
        raise
    except Exception:
      pass

  @staticmethod
  def load_dataframe(key: str) -> Optional[DataFrame]:
    """从磁盘加载 DataFrame"""
    cached_exists = DiskCache._check_exists(key, 'df')
    if cached_exists is False:
      return None

    file_path = CacheKey.make_file_path(key, 'df')
    if not file_path.exists():
      DiskCache._mark_exists(key, 'df', False)
      return None

    try:
      # 使用内存映射读取（多进程共享物理内存页）
      table = pq.read_table(file_path, memory_map=True)
      result = table.to_pandas()
      DiskCache._mark_exists(key, 'df', True)

      return result
    except Exception:
      try:
        file_path.unlink(missing_ok=True)
      except Exception:
        pass
      DiskCache._mark_exists(key, 'df', False)
      return None

  @staticmethod
  def save_pickle(key: str, data: Any) -> None:
    """保存 pickle 数据到磁盘（原子写入）"""
    file_path = CacheKey.make_file_path(key, 'pkl')
    try:
      serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

      fd, tmp_path = tempfile.mkstemp(suffix='.pkl', dir=file_path.parent)
      try:
        os.write(fd, serialized)
        os.close(fd)
        if file_path.exists():
          try:
            file_path.unlink()
          except Exception:
            pass
        os.replace(tmp_path, file_path)
        DiskCache._mark_exists(key, 'pkl', True)
      except Exception:
        try:
          os.close(fd)
        except Exception:
          pass
        try:
          os.unlink(tmp_path)
        except Exception:
          pass
        raise
    except Exception:
      pass

  @staticmethod
  def load_pickle(key: str) -> Optional[Any]:
    """从磁盘加载 pickle 数据"""
    cached_exists = DiskCache._check_exists(key, 'pkl')
    if cached_exists is False:
      return None

    file_path = CacheKey.make_file_path(key, 'pkl')
    if not file_path.exists():
      DiskCache._mark_exists(key, 'pkl', False)
      return None

    try:
      file_size = file_path.stat().st_size
      if file_size > 512 * 1024:  # 大于512KB使用mmap
        with open(file_path, 'rb') as f:
          with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            serialized = mm[:]
            result = pickle.loads(serialized)
      else:
        with open(file_path, 'rb') as f:
          serialized = f.read()
          result = pickle.loads(serialized)

      DiskCache._mark_exists(key, 'pkl', True)

      return result
    except Exception:
      try:
        file_path.unlink(missing_ok=True)
      except Exception:
        pass
      DiskCache._mark_exists(key, 'pkl', False)
      return None

  @staticmethod
  def clear_existence_cache():
    """清除文件存在性缓存"""
    with DiskCache._existence_lock:
      DiskCache._existence_cache.clear()

def cached_dataframe(method_name: str):
  """DataFrame 缓存装饰器（直接磁盘缓存，内存映射读取）"""

  def decorator(func):
    def wrapper(self, *args, **kwargs):
      cache_key = CacheKey.make_key(self.code, self.base_time, method_name, *args, **kwargs)

      # 直接从磁盘加载（内存映射，操作系统页面缓存）
      result = DiskCache.load_dataframe(cache_key)
      if result is not None:
        return result

      # 计算并缓存
      result = func(self, *args, **kwargs)
      if result is not None and not result.empty:
        DiskCache.save_dataframe(cache_key, result)
      return result

    return wrapper

  return decorator

def cached_value(method_name: str):
  """值缓存装饰器（直接磁盘缓存）"""

  def decorator(func):
    def wrapper(self, *args, **kwargs):
      cache_key = CacheKey.make_key(self.code, self.base_time, method_name, *args, **kwargs)

      result = DiskCache.load_pickle(cache_key)
      if result is not None:
        return result

      result = func(self, *args, **kwargs)
      if result is not None:
        DiskCache.save_pickle(cache_key, result)
      return result

    return wrapper

  return decorator

def cached_factor(factor_name: str):
  """因子计算缓存装饰器（直接磁盘缓存，自动函数代码哈希）"""

  def decorator(func):
    # 自动获取函数代码哈希
    func_hash = hash_function_code(func)

    def wrapper(self, ctx, *args, **kwargs):
      cache_key = CacheKey.make_key(ctx.code, ctx.base_time, f"factor_{factor_name}_h{func_hash}", *args, **kwargs)

      result = DiskCache.load_pickle(cache_key)
      if result is not None:
        return result

      result = func(self, ctx, *args, **kwargs)
      if result is not None:
        DiskCache.save_pickle(cache_key, result)
      return result

    return wrapper

  return decorator

def clear_cache():
  """清除所有缓存状态（文件缓存）"""
  DiskCache.clear_existence_cache()
  CacheKey.make_file_path.cache_clear()

def get_cache_stats() -> dict:
  """获取缓存统计信息"""
  return {
    'cache_dir': str(CACHE_DIR),
    'existence_cache_size': len(DiskCache._existence_cache),
  }
