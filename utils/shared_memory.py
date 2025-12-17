import pickle
import struct
from multiprocessing import shared_memory
from typing import Optional

import pandas as pd

class SharedMemoryCache:
  """支持多进程的共享内存缓存（零拷贝，无本地缓存）"""
  def __init__(self, cache_type: str):
    self.cache_type = cache_type  # 'daily' or 'minute'
    self._shm_registry: dict[str, shared_memory.SharedMemory] = {}  # 股票代码 -> SharedMemory对象

  def _get_shm_name(self, stock_code: str) -> str:
    """生成共享内存名称"""
    return f"wbr_cache_{self.cache_type}_{stock_code.replace('.', '_')}"

  def put(self, stock_code: str, data: Optional[pd.DataFrame]) -> None:
    """存入缓存（仅在主进程调用）"""
    if data is None:
      return

    # 序列化DataFrame
    serialized = pickle.dumps(data)
    data_size = len(serialized)

    # 创建共享内存：4字节头（存储数据大小） + 实际数据
    shm_name = self._get_shm_name(stock_code)
    total_size = 4 + data_size

    try:
      # 如果已存在，先释放
      if stock_code in self._shm_registry:
        old_shm = self._shm_registry[stock_code]
        old_shm.close()
        old_shm.unlink()

      # 创建新的共享内存
      shm = shared_memory.SharedMemory(create=True, size=total_size, name=shm_name)

      # 写入数据大小（4字节）
      shm.buf[0:4] = struct.pack('I', data_size)
      # 写入序列化数据
      shm.buf[4:4+data_size] = serialized

      self._shm_registry[stock_code] = shm

    except Exception as e:
      # 共享内存创建失败，静默失败（子进程会回退到正常加载）
      pass

  def get(self, stock_code: str) -> Optional[pd.DataFrame]:
    """从缓存获取（主进程和子进程都可调用）

    每次都从共享内存反序列化，不使用本地缓存。
    这样可以避免子进程内存累积，适合只读场景。
    """
    shm_name = self._get_shm_name(stock_code)
    shm = None

    try:
      # 连接共享内存
      shm = shared_memory.SharedMemory(name=shm_name)

      # 读取数据大小
      data_size = struct.unpack('I', bytes(shm.buf[0:4]))[0]

      # 读取序列化数据并反序列化
      serialized = bytes(shm.buf[4:4+data_size])
      data = pickle.loads(serialized)

      # 记录共享内存引用（主进程用于后续清理）
      if stock_code not in self._shm_registry:
        self._shm_registry[stock_code] = shm
      else:
        # 子进程：立即关闭连接
        shm.close()

      return data

    except FileNotFoundError:
      # 共享内存不存在，返回None
      if shm:
        shm.close()
      return None
    except Exception:
      # 其他错误，返回None
      if shm:
        try:
          shm.close()
        except:
          pass
      return None

  def contains(self, stock_code: str) -> bool:
    """检查缓存是否包含该股票"""
    shm_name = self._get_shm_name(stock_code)
    try:
      shm = shared_memory.SharedMemory(name=shm_name)
      shm.close()
      return True
    except FileNotFoundError:
      return False

  def cleanup(self) -> None:
    """清理所有共享内存（仅在主进程退出时调用）"""
    for stock_code, shm in self._shm_registry.items():
      try:
        shm.close()
        shm.unlink()
      except:
        pass
    self._shm_registry.clear()
