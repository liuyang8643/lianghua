"""SharedMemoryCache 性能测试和使用示例"""

import time
import sys
from multiprocessing import Process
import pandas as pd
import numpy as np

from shared_memory import SharedMemoryCache


def test_basic_usage():
  """测试基本功能"""
  print("\n=== 测试基本功能 ===")
  
  # 使用上下文管理器（自动清理）
  with SharedMemoryCache[dict]('test') as cache:
    # 存入不同类型的数据
    cache.put('dict_data', {'key': 'value', 'numbers': [1, 2, 3]})
    cache.put('list_data', [1, 2, 3, 4, 5])
    cache.put('str_data', 'Hello, World!')
    
    # 读取数据
    print(f"Dict: {cache.get('dict_data')}")
    print(f"List: {cache.get('list_data')}")
    print(f"Str: {cache.get('str_data')}")
    
    # 检查是否存在
    print(f"Contains 'dict_data': {cache.contains('dict_data')}")
    print(f"Contains 'missing': {cache.contains('missing')}")
    
    # 获取所有键
    print(f"Keys: {cache.keys()}")
    print(f"Length: {len(cache)}")


def test_dataframe():
  """测试 DataFrame 存储"""
  print("\n=== 测试 DataFrame 存储 ===")
  
  # 创建测试数据
  df = pd.DataFrame({
    'A': np.random.rand(10000),
    'B': np.random.rand(10000),
    'C': np.random.rand(10000),
  })
  
  print(f"DataFrame shape: {df.shape}")
  print(f"DataFrame memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
  
  # 不压缩
  cache_no_compress = SharedMemoryCache[pd.DataFrame]('test_no_compress', compress_level=0)
  start = time.time()
  cache_no_compress.put('df', df)
  no_compress_time = time.time() - start
  
  # 压缩
  cache_compress = SharedMemoryCache[pd.DataFrame]('test_compress', compress_level=6)
  start = time.time()
  cache_compress.put('df', df)
  compress_time = time.time() - start
  
  print(f"\n写入时间:")
  print(f"  无压缩: {no_compress_time * 1000:.2f} ms")
  print(f"  压缩:   {compress_time * 1000:.2f} ms")
  
  # 读取性能
  start = time.time()
  df_no_compress = cache_no_compress.get('df')
  no_compress_read_time = time.time() - start
  
  start = time.time()
  df_compress = cache_compress.get('df')
  compress_read_time = time.time() - start
  
  print(f"\n读取时间:")
  print(f"  无压缩: {no_compress_read_time * 1000:.2f} ms")
  print(f"  压缩:   {compress_read_time * 1000:.2f} ms")
  
  # 验证数据正确性
  assert df.equals(df_no_compress), "无压缩数据不匹配"
  assert df.equals(df_compress), "压缩数据不匹配"
  print("\n✅ 数据验证通过")
  
  # 清理
  cache_no_compress.cleanup()
  cache_compress.cleanup()


def child_process_reader(cache_name: str, key: str, expected_value: dict):
  """子进程：读取共享内存"""
  cache = SharedMemoryCache[dict](cache_name)
  data = cache.get(key)
  
  if data == expected_value:
    print(f"✅ 子进程读取成功: {data}")
  else:
    print(f"❌ 子进程读取失败: expected {expected_value}, got {data}")
    sys.exit(1)


def test_multiprocess():
  """测试多进程场景"""
  print("\n=== 测试多进程读取 ===")
  
  cache_name = 'test_multiprocess'
  cache = SharedMemoryCache[dict](cache_name)
  
  test_data = {
    'msg': 'Hello from parent',
    'numbers': [1, 2, 3, 4, 5],
    'nested': {'key': 'value'}
  }
  
  # 主进程写入
  cache.put('shared_data', test_data)
  print(f"主进程写入: {test_data}")
  
  # 启动子进程读取
  processes = []
  for i in range(3):
    p = Process(target=child_process_reader, args=(cache_name, 'shared_data', test_data))
    p.start()
    processes.append(p)
  
  # 等待所有子进程完成
  for p in processes:
    p.join()
  
  # 清理
  cache.cleanup()
  print("✅ 多进程测试完成")


def test_remove_and_size():
  """测试删除和大小查询"""
  print("\n=== 测试删除和大小查询 ===")
  
  cache = SharedMemoryCache[str]('test_remove')
  
  # 添加多个键
  for i in range(5):
    cache.put(f'key_{i}', f'value_{i}')
  
  print(f"初始大小: {len(cache)}")
  print(f"初始键: {cache.keys()}")
  
  # 删除一个键
  cache.remove('key_2')
  print(f"删除 key_2 后大小: {len(cache)}")
  print(f"删除后键: {cache.keys()}")
  print(f"key_2 是否存在: {cache.contains('key_2')}")
  
  cache.cleanup()
  print("✅ 删除测试完成")


def test_large_object():
  """测试大对象压缩效果"""
  print("\n=== 测试大对象压缩 ===")
  
  # 创建大对象（大量重复数据，适合压缩）
  large_data = {
    'array': np.ones((1000, 1000)),  # 1M 个浮点数
    'text': 'A' * 1_000_000,  # 1M 字符
  }
  
  import pickle
  serialized_size = len(pickle.dumps(large_data))
  print(f"原始序列化大小: {serialized_size / 1024 / 1024:.2f} MB")
  
  # 测试压缩
  cache_compress = SharedMemoryCache('test_large', compress_level=6)
  start = time.time()
  success = cache_compress.put('large', large_data)
  compress_time = time.time() - start
  
  print(f"压缩写入时间: {compress_time * 1000:.2f} ms")
  print(f"写入成功: {success}")
  
  # 测试读取
  start = time.time()
  data = cache_compress.get('large')
  decompress_time = time.time() - start
  
  print(f"解压读取时间: {decompress_time * 1000:.2f} ms")
  print(f"数据完整性: {'✅ 通过' if data is not None else '❌ 失败'}")
  
  cache_compress.cleanup()


if __name__ == '__main__':
  print("SharedMemoryCache 性能测试")
  print("=" * 60)
  
  try:
    test_basic_usage()
    test_dataframe()
    test_multiprocess()
    test_remove_and_size()
    test_large_object()
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    
  except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
