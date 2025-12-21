"""
哈希工具函数
提供函数代码自动哈希功能
"""
import hashlib
import inspect
from functools import lru_cache
from typing import Callable

@lru_cache(maxsize=32)
def hash_function_code(func: Callable) -> str:
  """
  获取函数所在文件的完整源代码并生成哈希值

  Args:
    func: 要哈希的函数对象

  Returns:
    8位十六进制哈希字符串
  """
  try:
    # 获取函数所在的源文件路径
    source_file = inspect.getsourcefile(func)
    if source_file:
      # 读取整个文件内容
      with open(source_file, 'r', encoding='utf-8') as f:
        file_content = f.read()
      # 使用 blake2b 生成紧凑哈希（4字节=8位十六进制）
      code_hash = hashlib.blake2b(file_content.encode('utf-8'), digest_size=4).hexdigest()
      return code_hash
    else:
      raise OSError("Cannot locate source file")
  except (OSError, TypeError, IOError) as e:
    # 如果无法获取源文件（例如内置函数），使用函数名和模块名作为fallback
    fallback_str = f"{func.__module__}.{func.__qualname__}"
    code_hash = hashlib.blake2b(fallback_str.encode('utf-8'), digest_size=4).hexdigest()
    return code_hash

def hash_string(s: str) -> str:
  """
  对字符串生成哈希值

  Args:
    s: 输入字符串

  Returns:
    8位十六进制哈希字符串
  """
  return hashlib.blake2b(s.encode('utf-8'), digest_size=4).hexdigest()