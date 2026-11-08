""" 启动QMT进程 """

import os
import shutil
import psutil
import subprocess
from typing import Optional, Union

from configs.env import QMT_ROOT_DIR

def get_qmt_process() -> Optional[psutil.Process]:
  """获取 QMT 进程对象"""
  expected = {os.path.normcase(os.path.realpath(os.path.join(path, 'XtMiniQmt.exe')))
              for path in QMT_ROOT_DIR}
  for proc in psutil.process_iter(['pid', 'name']):
    try:
      if (proc.info['name'] == 'XtMiniQmt.exe'
          and os.path.normcase(os.path.realpath(proc.exe())) in expected):
        return proc
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
      pass
  return None

def find_valid_qmt_dir():
  """保留旧版 linkMini 启动方式；新版直接使用客户端自己的登录配置。"""
  for qmt_dir in QMT_ROOT_DIR:
    src = os.path.join(qmt_dir, '_linkMini')
    exe = os.path.join(qmt_dir, 'XtMiniQmt.exe')

    if os.path.isfile(exe):
      if os.path.isfile(src):
        dst = os.path.join(qmt_dir, 'linkMini')
        shutil.copy(src, dst)
      return exe

  # 如果没有找到有效目录，抛出异常
  raise FileNotFoundError('未找到配置的 XtMiniQmt.exe；请安装并登录券商客户端')

def start_qmt() -> Union[subprocess.Popen, psutil.Process]:
  """启动 QMT 交易端，如果已运行则返回现有进程"""
  # 检查是否已有进程在运行
  existing_process = get_qmt_process()
  if existing_process:
    print(f"QMT 程序已在运行，进程ID: {existing_process.pid}")
    return existing_process

  # 查找有效的 QMT 目录，执行复制操作，获取可执行文件路径
  exe_path = find_valid_qmt_dir()
  command = [exe_path]
  if os.path.isfile(os.path.join(os.path.dirname(exe_path), 'linkMini')):
    command.append('linkMini')
  process = psutil.Popen(command, cwd=os.path.dirname(exe_path))
  print(f"QMT 程序启动成功，进程ID: {process.pid}")
  return process

if __name__ == "__main__":
  start_qmt()
