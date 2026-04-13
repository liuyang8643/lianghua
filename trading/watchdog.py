import argparse
import os
import signal
import subprocess
import time
import sys
import datetime
from typing import Optional
from trading.qmt import start_qmt
from utils.stock.time import is_trading_day
from utils.sys import terminate_process_tree

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
MAIN_MODULE = 'trading.main'
DEFAULT_INDIVIDUAL_CONFIG = os.path.join(REPO_ROOT, 'configs', 'best_individual_config.json')


def _resolve_path(path: str) -> str:
  if os.path.isabs(path):
    return path
  return os.path.abspath(os.path.join(REPO_ROOT, path))


def monitor_main(individual_config: str):
  """
  监控并管理QMT交易程序的主函数
  :param individual_config: Individual_config JSON 文件路径
  """
  resolved_individual_config = _resolve_path(individual_config)
  terminate_flag = False
  restart_delay = 10  # 初始重启延迟（秒）
  process_start_time = datetime.time(9, 0)  # 进程启动时间（开盘前）
  process_end_time = datetime.time(16, 0)  # 进程结束时间（收盘后）
  qmt_process: Optional[subprocess.Popen] = None
  main_process: Optional[subprocess.Popen] = None

  def handle_sigterm(signum, frame):
    """处理终止信号的函数"""
    nonlocal terminate_flag
    terminate_flag = True
    print("收到终止信号(SIGTERM)，准备退出程序...")
    stop_processes()  # 先停止子进程再退出
    sys.exit(0)

  signal.signal(signal.SIGTERM, handle_sigterm)

  def stop_processes():
    """停止所有子进程"""
    nonlocal qmt_process, main_process

    # 使用新的 terminate_process_tree 函数优雅地终止进程
    if main_process:
      terminate_process_tree(main_process)
      main_process = None

    if qmt_process:
      terminate_process_tree(qmt_process)
      qmt_process = None

  def start_processes():
    """启动QMT和主交易进程"""
    nonlocal qmt_process, main_process, restart_delay
    try:
      print("正在启动QMT平台...")
      qmt_process = start_qmt()
      print(f"QMT平台已启动 (进程ID: {qmt_process.pid})")
      time.sleep(5)  # 等待QMT初始化
      print(f"正在启动主交易进程，配置文件: {resolved_individual_config}")
      main_command = [
        sys.executable,
        '-m',
        MAIN_MODULE,
        '--individual-config',
        resolved_individual_config,
      ]
      main_process = subprocess.Popen(main_command)
      print(f"主交易进程已启动 (进程ID: {main_process.pid})")
      restart_delay = 10  # 重置重启延迟
    except Exception as setup_e:
      print(f"启动进程时出错: {setup_e}")
      # 确保清理任何可能已启动的进程
      stop_processes()
      # 短暂延迟后重试
      time.sleep(10)

  try:
    while True:
      now = datetime.datetime.now()
      current_time = now.time()

      # 检查QMT进程状态
      if qmt_process and qmt_process.poll() is not None:
        print(f"QMT进程已退出 (返回码: {qmt_process.returncode})，需要重启所有进程")
        stop_processes()
        time.sleep(restart_delay)
        restart_delay = min(2 * restart_delay, 300)  # 最大延迟改为5分钟
        continue

      # 检查是否应该停止进程（收盘后）
      if process_end_time < current_time and (main_process or qmt_process):
        print("市场已收盘，停止所有进程...")
        stop_processes()
      # 检查是否应该启动进程（开盘前且是交易日）
      elif is_trading_day(now.date()) and process_start_time < current_time < process_end_time and not (main_process and qmt_process):
        print("市场交易时段，启动交易程序...")
        start_processes()
      # 检查主进程是否异常退出
      elif main_process and main_process.poll() is not None:
        if terminate_flag or main_process.returncode == signal.SIGTERM:
          print("主进程正常终止，退出监控...")
          break
        print(f"主进程异常退出 (返回码: {main_process.returncode})，将在 {restart_delay} 秒后重启...")
        stop_processes()  # 确保两个进程都被停止
        time.sleep(restart_delay)
        restart_delay = min(2 * restart_delay, 300)  # 最大延迟改为5分钟
        if current_time < process_end_time and is_trading_day(now.date()):
          start_processes()

      # 当前非交易时段，休眠一段时间再检查
      time.sleep(5)

  except KeyboardInterrupt:
    print("收到键盘中断，正在停止所有进程...")
    stop_processes()
    print("已停止所有进程")
    sys.exit(0)
  except Exception as e:
    print(f"监控过程中发生异常: {e}")
    stop_processes()
    sys.exit(1)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--individual-config',
    type=str,
    default=DEFAULT_INDIVIDUAL_CONFIG,
    help='Individual_config JSON文件路径'
  )
  args = parser.parse_args()

  print('QMT Watchdog 启动，开始监控主程序...')
  print(f'使用配置文件: {_resolve_path(args.individual_config)}')
  monitor_main(args.individual_config)
