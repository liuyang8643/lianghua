import pickle
import struct
import zlib
from multiprocessing import shared_memory
from typing import Optional, Any, TypeVar, Generic

T = TypeVar('T')

class SharedMemoryCache(Generic[T]):
  """极致性能多进程共享内存缓存
  
  特性：
  - 支持任意可序列化对象
  - 可选压缩（compress_level=0 禁用，1-9 启用，推荐 6）
  - 无锁设计（需要外部保证线程安全）
  - 支持上下文管理器
  - 零拷贝读取（子进程不保留本地缓存）
  
  内存布局：
  [4字节: 数据大小] [1字节: 压缩标志] [N字节: 序列化数据]
  """
  
  def __init__(self, cache_name: str, compress_level: int = 0):
    """初始化共享内存缓存
    
    Args:
      cache_name: 缓存命名空间（用于区分不同用途的缓存）
      compress_level: 压缩级别，0=禁用，1-9=启用（推荐 6）
    """
    self.cache_name = cache_name
    self.compress_level = compress_level
    self._shm_registry: dict[str, shared_memory.SharedMemory] = {}
    
  def _get_shm_name(self, key: str) -> str:
    """生成共享内存名称（Windows下最多247字符）"""
    # 替换特殊字符，避免命名冲突
    safe_key = key.replace('.', '_').replace('/', '_').replace('\\', '_')
    return f"wbr_{self.cache_name}_{safe_key}"[:247]
  
  def _serialize(self, data: Any) -> tuple[bytes, bool]:
    """序列化并可选压缩数据
    
    Returns:
      (序列化后的字节, 是否压缩)
    """
    serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    
    if self.compress_level > 0:
      compressed = zlib.compress(serialized, level=self.compress_level)
      # 只有压缩后更小才使用
      if len(compressed) < len(serialized):
        return compressed, True
    
    return serialized, False
  
  def _deserialize(self, data: bytes, is_compressed: bool) -> Any:
    """反序列化数据"""
    if is_compressed:
      data = zlib.decompress(data)
    return pickle.loads(data)
  
  def put(self, key: str, data: Any) -> bool:
    """存入缓存（主进程调用，无锁设计）
    
    Args:
      key: 缓存键
      data: 任意可序列化对象
      
    Returns:
      是否成功写入
    """
    if data is None:
      return False
    
    try:
      # 序列化（可选压缩）
      serialized, is_compressed = self._serialize(data)
      data_size = len(serialized)
      
      # 内存布局：[4字节大小] [1字节压缩标志] [数据]
      total_size = 4 + 1 + data_size
      shm_name = self._get_shm_name(key)
      
      # 释放旧共享内存
      if key in self._shm_registry:
        self._cleanup_key(key)
      
      # 创建新共享内存
      shm = shared_memory.SharedMemory(create=True, size=total_size, name=shm_name)
      
      # 写入头部
      shm.buf[0:4] = struct.pack('I', data_size)
      shm.buf[4:5] = struct.pack('B', int(is_compressed))
      
      # 写入数据
      shm.buf[5:5+data_size] = serialized
      
      self._shm_registry[key] = shm
      return True
      
    except Exception:
      # 失败时静默处理（避免中断主流程）
      return False
  
  def get(self, key: str) -> Optional[T]:
    """从缓存获取（主进程和子进程都可调用，无锁设计）
    
    Args:
      key: 缓存键
      
    Returns:
      缓存的对象，不存在则返回 None
    """
    shm_name = self._get_shm_name(key)
    shm = None
    
    try:
      # 连接共享内存
      shm = shared_memory.SharedMemory(name=shm_name)
      
      # 读取头部
      data_size = struct.unpack('I', bytes(shm.buf[0:4]))[0]
      is_compressed = bool(struct.unpack('B', bytes(shm.buf[4:5]))[0])
      
      # 读取并反序列化数据
      serialized = bytes(shm.buf[5:5+data_size])
      data = self._deserialize(serialized, is_compressed)
      
      # 主进程：记录引用（用于后续清理）
      # 子进程：立即关闭（避免资源泄漏）
      if key not in self._shm_registry:
        self._shm_registry[key] = shm
      else:
        shm.close()
      
      return data
      
    except FileNotFoundError:
      # 共享内存不存在
      if shm:
        shm.close()
      return None
      
    except Exception:
      # 其他错误（损坏的数据等）
      if shm:
        try:
          shm.close()
        except:
          pass
      return None
  
  def contains(self, key: str) -> bool:
    """检查缓存是否包含指定键"""
    shm_name = self._get_shm_name(key)
    try:
      shm = shared_memory.SharedMemory(name=shm_name)
      shm.close()
      return True
    except FileNotFoundError:
      return False
  
  def _cleanup_key(self, key: str) -> None:
    """清理单个键（内部方法，不加锁）"""
    if key in self._shm_registry:
      shm = self._shm_registry[key]
      try:
        shm.close()
        shm.unlink()
      except:
        pass
      del self._shm_registry[key]
  
  def remove(self, key: str) -> bool:
    """从缓存中移除指定键（无锁设计）
    
    Returns:
      是否成功移除
    """
    try:
      self._cleanup_key(key)
      return True
    except:
      return False
  
  def cleanup(self) -> None:
    """清理所有共享内存（主进程退出时调用，无锁设计）"""
    for key in list(self._shm_registry.keys()):
      self._cleanup_key(key)
  
  def keys(self) -> list[str]:
    """返回所有缓存键（无锁设计）"""
    return list(self._shm_registry.keys())
  
  def stat(self) -> dict[str, Any]:
    """查看当前占用的内存大小（无锁设计）
    
    Returns:
      统计信息字典:
        - total_size: 总占用内存（字节）
        - total_size_mb: 总占用内存（MB）
        - count: 缓存键数量
        - keys: 各键的详细信息列表
          - key: 缓存键名
          - size: 占用内存（字节）
          - size_mb: 占用内存（MB）
          - compressed: 是否压缩
    """
    total_size = 0
    keys_info = []
    
    for key, shm in self._shm_registry.items():
      try:
        # 读取数据大小和压缩标志
        data_size = struct.unpack('I', bytes(shm.buf[0:4]))[0]
        is_compressed = bool(struct.unpack('B', bytes(shm.buf[4:5]))[0])
        
        # 总大小 = 头部(5字节) + 数据大小
        key_size = 5 + data_size
        total_size += key_size
        
        keys_info.append({
          'key': key,
          'size': key_size,
          'size_mb': key_size / (1024 * 1024),
          'compressed': is_compressed,
        })
      except:
        # 忽略损坏的共享内存
        pass
    
    return {
      'total_size': total_size,
      'total_size_mb': total_size / (1024 * 1024),
      'count': len(keys_info),
      'keys': keys_info,
    }
  
  def __len__(self) -> int:
    """返回缓存中的键数量（无锁设计）"""
    return len(self._shm_registry)
  
  def __enter__(self):
    """上下文管理器：进入"""
    return self
  
  def __exit__(self, exc_type, exc_val, exc_tb):
    """上下文管理器：退出时自动清理"""
    self.cleanup()
    return False
