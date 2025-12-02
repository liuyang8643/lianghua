""" 启动QMT进程 """

import os
import shutil
import psutil
import subprocess
from typing import Optional, Union

from configs import QMT_ROOT_DIR

def get_qmt_process() -> Optional[psutil.Process]:
  """获取 QMT 进程对象"""
  for proc in psutil.process_iter(['pid', 'name']):
    try:
      if proc.info['name'] == 'XtMiniQmt.exe':
        return proc
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
      pass
  return None

def find_valid_qmt_dir():
  """查找有效的 QMT 目录，执行复制操作，并返回可执行文件路径"""
  for qmt_dir in QMT_ROOT_DIR:
    src = os.path.join(qmt_dir, '_linkMini')
    exe = os.path.join(qmt_dir, 'XtMiniQmt.exe')

    if os.path.exists(src) and os.path.exists(exe):
      dst = os.path.join(qmt_dir, 'linkMini')
      shutil.copy(src, dst)
      return exe

  # 如果没有找到有效目录，抛出异常
  raise FileNotFoundError(f"未找到包含 _linkMini 文件和 XtMiniQmt.exe 的 QMT 目录！")

def start_qmt() -> Union[subprocess.Popen, psutil.Process]:
  """启动 QMT 交易端，如果已运行则返回现有进程"""
  # 检查是否已有进程在运行
  existing_process = get_qmt_process()
  if existing_process:
    print(f"QMT 程序已在运行，进程ID: {existing_process.pid}")
    return existing_process

  # 查找有效的 QMT 目录，执行复制操作，获取可执行文件路径
  exe_path = find_valid_qmt_dir()
  process = psutil.Popen([exe_path, 'linkMini'])
  print(f"QMT 程序启动成功，进程ID: {process.pid}")
  return process

if __name__ == "__main__":
  start_qmt()
