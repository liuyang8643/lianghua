import os
import subprocess
from typing import Union

import psutil

def terminate_process_tree(
    process: Union[int, psutil.Process, subprocess.Popen, None] = None,
    timeout: int = 10,
):
  """
  优雅地结束指定进程及其子进程。
  :param process: 进程对象、进程ID或None（默认当前进程）
  :param timeout: 等待进程终止的超时时间（秒）
  """
  if process is None:
    pid = os.getpid()  # 获取当前进程 ID
    process_obj = psutil.Process(pid)
  elif isinstance(process, int):
    pid = process
    process_obj = psutil.Process(pid)
  elif isinstance(process, subprocess.Popen):
    pid = process.pid
    process_obj = psutil.Process(pid)
  elif isinstance(process, psutil.Process):
    pid = process.pid
    process_obj = process
  else:
    # 检查是否是 ProcessWrapper 类型的对象
    if hasattr(process, 'pid') and hasattr(process, 'poll'):
      pid = process.pid
      process_obj = psutil.Process(pid)
    else:
      raise ValueError(f"不支持的进程类型: {type(process)}")

  try:
    # 获取进程名称
    process_name = process_obj.name()
    print(f"正在终止进程: {process_name} (PID: {pid})")

    # 检查进程是否还在运行
    if not process_obj.is_running():
      print(f"进程 {process_name} 已经终止")
      return

    # 如果是 subprocess.Popen 对象，先尝试使用其 poll() 方法
    if isinstance(process, subprocess.Popen):
      if process.poll() is None:
        print(f"正在优雅终止 {process_name}...")
        process.terminate()
        try:
          process.wait(timeout=timeout)
          print(f"进程 {process_name} 已成功终止")
          return
        except subprocess.TimeoutExpired:
          print(f"进程 {process_name} 未响应，强制终止")
          process.kill()
          print(f"进程 {process_name} 已强制终止")
          return
      else:
        print(f"进程 {process_name} 已经终止")
        return

    # 对于其他类型的进程对象，使用 psutil 处理
    # 先终止所有子进程
    children = process_obj.children(recursive=True)
    if children:
      print(f"正在终止 {len(children)} 个子进程...")
      for child in children:
        try:
          child_name = child.name()
          print(f"\t终止子进程: {child_name} (PID: {child.pid})")
          child.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
          pass

      # 等待子进程终止
      psutil.wait_procs(children, timeout=timeout)

    # 终止主进程
    print(f"正在优雅终止主进程 {process_name}...")
    process_obj.terminate()

    try:
      process_obj.wait(timeout=timeout)
      print(f"进程 {process_name} 已成功终止")
    except psutil.TimeoutExpired:
      print(f"进程 {process_name} 未响应，强制终止")
      try:
        process_obj.kill()
        process_obj.wait(timeout=5)
        print(f"进程 {process_name} 已强制终止")
      except (psutil.NoSuchProcess, psutil.AccessDenied):
        print(f"进程 {process_name} 可能已被系统终止")

  except psutil.NoSuchProcess:
    print(f"进程 (PID: {pid}) 不存在或已终止")
  except psutil.AccessDenied:
    print(f"没有权限访问进程 (PID: {pid})")
  except Exception as e:
    print(f"终止进程时发生错误: {e}")
