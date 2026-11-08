"""24x7 守护 QMT 与自动实盘主进程。

watchdog 只拥有自己创建的资源：主交易进程始终由它创建；QMT 若在启动前已经
存在则只监控、不终止。日内阶段和跨日重置均由 ``trading.scheduler`` 负责，
watchdog 不在 16:00 强制结束进程，以免打断盘后对账或全量数据更新。
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from typing import Callable, Optional

import psutil
from filelock import FileLock, Timeout

from trading.qmt import get_qmt_process, start_qmt
from utils.sys import terminate_process_tree


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
MAIN_MODULE = "trading.main"
DEFAULT_INDIVIDUAL_CONFIG = os.path.join(REPO_ROOT, "configs", "config.json")
WATCHDOG_LOCK_PATH = os.path.join(
  REPO_ROOT, "data", "live_trades", "trading_watchdog.lock")
WATCHDOG_HALT_PATH = os.path.join(
  REPO_ROOT, "data", "live_trades", "trading_watchdog.halt")

POLL_INTERVAL_SEC = 5.0
QMT_STARTUP_WAIT_SEC = 5.0
INITIAL_RESTART_DELAY_SEC = 10.0
MAX_RESTART_DELAY_SEC = 300.0
STABLE_PROCESS_RESET_SEC = 300.0


class WatchdogAlreadyRunning(RuntimeError):
  """另一个 watchdog 已持有单实例锁。"""


def _resolve_path(path: str) -> str:
  if os.path.isabs(path):
    return path
  return os.path.abspath(os.path.join(REPO_ROOT, path))


def _process_returncode(process) -> Optional[int]:
  """统一读取 ``Popen`` / ``psutil.Process`` 的退出状态。

  返回 ``None`` 表示仍存活；``psutil.Process`` 没有真实 return code，退出时返回
  ``-1``。先检查 ``poll``，因此也兼容 ``psutil.Popen`` 和测试替身。
  """
  if process is None:
    return -1

  try:
    poll = getattr(process, "poll", None)
    if callable(poll):
      return poll()

    is_running = getattr(process, "is_running", None)
    if not callable(is_running) or not is_running():
      return -1
    status = getattr(process, "status", None)
    if callable(status) and status() == psutil.STATUS_ZOMBIE:
      return -1
    return None
  except (psutil.NoSuchProcess, psutil.ZombieProcess):
    return -1
  except psutil.AccessDenied:
    # 无权读取状态不等于已退出；避免因权限瞬态误杀正常实盘。
    return None


def _is_running(process) -> bool:
  return _process_returncode(process) is None


def _was_spawned_by_start_qmt(process) -> bool:
  """区分 start_qmt 的新进程与竞态中发现的既有 QMT。

  ``start_qmt`` 新建时返回 ``psutil.Popen``（有 ``poll``），复用既有实例时返回
  ``psutil.Process``（无 ``poll``）。只把前者标记为 watchdog 所有。
  """
  return callable(getattr(process, "poll", None))


def _acquire_watchdog_lock(lock_path: str = WATCHDOG_LOCK_PATH) -> FileLock:
  os.makedirs(os.path.dirname(lock_path), exist_ok=True)
  lock = FileLock(lock_path)
  try:
    lock.acquire(timeout=0)
  except Timeout as exc:
    raise WatchdogAlreadyRunning(
      f"已有一个实盘 watchdog 运行中（锁: {lock_path}）") from exc
  return lock


def _clear_halt_marker(halt_path: str = WATCHDOG_HALT_PATH):
  """人工重新启动 watchdog 时清除上一次紧急制动标记。"""
  try:
    os.unlink(halt_path)
  except FileNotFoundError:
    return
  print(f"已清除紧急制动标记: {halt_path}", flush=True)


class TradingWatchdog:
  """保持 QMT 与 ``trading.main`` 存活，不接管 scheduler 的日内时序。"""

  def __init__(
      self,
      individual_config: str,
      *,
      qmt_finder: Callable = get_qmt_process,
      qmt_starter: Callable = start_qmt,
      popen_factory: Callable = subprocess.Popen,
      terminate_func: Callable = terminate_process_tree,
      sleep_func: Callable[[float], None] = time.sleep,
      monotonic_func: Callable[[], float] = time.monotonic,
      poll_interval: float = POLL_INTERVAL_SEC,
      qmt_startup_wait: float = QMT_STARTUP_WAIT_SEC,
      halt_path: str = WATCHDOG_HALT_PATH,
      trading_backend: str = 'bridge',
  ):
    self.individual_config = _resolve_path(individual_config)
    self.qmt_finder = qmt_finder
    self.qmt_starter = qmt_starter
    self.popen_factory = popen_factory
    self.terminate_func = terminate_func
    self.sleep = sleep_func
    self.monotonic = monotonic_func
    self.poll_interval = poll_interval
    self.qmt_startup_wait = qmt_startup_wait
    self.halt_path = halt_path
    self.trading_backend = trading_backend

    self.qmt_process = None
    self.qmt_owned = False
    self.main_process = None
    self.main_started_at: Optional[float] = None
    self.restart_delay = INITIAL_RESTART_DELAY_SEC
    self.stop_requested = False

  def request_stop(self):
    self.stop_requested = True

  def _emergency_halt_requested(self) -> bool:
    return os.path.exists(self.halt_path)

  def _start_or_attach_qmt(self) -> bool:
    """确保 QMT 存活；返回是否由本 watchdog 新创建。"""
    existing = self.qmt_finder()
    if _is_running(existing):
      self.qmt_process = existing
      self.qmt_owned = False
      print(f"复用既有 QMT (PID={existing.pid})，watchdog 不拥有该进程", flush=True)
      return False

    print("正在启动 QMT...", flush=True)
    process = self.qmt_starter()
    if not _is_running(process):
      raise RuntimeError("QMT 启动后立即退出")
    self.qmt_process = process
    self.qmt_owned = _was_spawned_by_start_qmt(process)
    ownership = "watchdog 创建" if self.qmt_owned else "竞态中发现的既有实例"
    print(f"QMT 已就绪 (PID={process.pid}, {ownership})", flush=True)
    return self.qmt_owned

  def _start_main(self):
    command = [
      sys.executable,
      '-X', 'utf8',
      "-m",
      MAIN_MODULE,
      "--individual-config",
      self.individual_config,
    ]
    command += ['--trading-backend', self.trading_backend]
    print(f"正在启动自动实盘: {self.individual_config}", flush=True)
    self.main_process = self.popen_factory(command, cwd=REPO_ROOT)
    self.main_started_at = self.monotonic()
    print(f"自动实盘已启动 (PID={self.main_process.pid})", flush=True)

  def _stop_main(self, reason: str) -> bool:
    """终止旧主进程并确认结果；仍存活时保留引用，返回 ``False``。"""
    process = self.main_process
    if not _is_running(process):
      self.main_process = None
      self.main_started_at = None
      return True
    print(f"终止 watchdog 子交易进程 (PID={process.pid}): {reason}", flush=True)
    self.terminate_func(process)
    if _is_running(process):
      # terminate_process_tree 会记录并吞掉部分系统异常；必须以进程实际状态为准。
      print(
        f"旧主进程 PID={process.pid} 终止未生效；保留引用并禁止启动新主进程",
        flush=True,
      )
      return False
    self.main_process = None
    self.main_started_at = None
    return True

  def _stop_owned_qmt(self):
    process = self.qmt_process
    owned = self.qmt_owned
    if not owned:
      if _is_running(process):
        print(f"保留启动前已存在的 QMT (PID={process.pid})", flush=True)
      self.qmt_process = None
      self.qmt_owned = False
      return
    if _is_running(process):
      print(f"终止 watchdog 创建的 QMT (PID={process.pid})", flush=True)
      # 终止失败时保留所有权与引用，绝不把它误判成外部 QMT。
      self.terminate_func(process)
    self.qmt_process = None
    self.qmt_owned = False

  def _backoff(self, reason: str):
    delay = self.restart_delay
    print(f"{reason}；{delay:.0f}s 后重试", flush=True)
    self.sleep(delay)
    self.restart_delay = min(delay * 2, MAX_RESTART_DELAY_SEC)

  def _reset_backoff_if_stable(self):
    if self.main_started_at is None:
      return
    if self.monotonic() - self.main_started_at < STABLE_PROCESS_RESET_SEC:
      return
    self.restart_delay = INITIAL_RESTART_DELAY_SEC
    # 清掉时间戳，避免每轮重复执行；下次重新启动会重新设置。
    self.main_started_at = None

  def run_once(self):
    """执行一次守护检查，拆出供无真实进程的单元测试使用。"""
    if self._emergency_halt_requested():
      print(f"检测到紧急制动标记，停止自动拉起: {self.halt_path}", flush=True)
      self.request_stop()
      return

    qmt_exit = _process_returncode(self.qmt_process)
    if self.qmt_process is None or qmt_exit is not None:
      if self.qmt_process is not None:
        print(f"QMT 已退出 (状态={qmt_exit})", flush=True)
        # 主进程持有旧连接，必须只重启自己的主进程；不能碰外部 QMT。
        if not self._stop_main("QMT 已退出，需重建交易连接"):
          # 保留死亡 QMT 的引用，使下一轮继续命中该分支并重试终止旧 main。
          self._backoff("旧主进程仍存活，暂不启动新 QMT/main")
          return
        self.qmt_process = None
        self.qmt_owned = False
      try:
        created = self._start_or_attach_qmt()
        if created and self.qmt_startup_wait > 0:
          self.sleep(self.qmt_startup_wait)
      except Exception as exc:
        self._backoff(f"QMT 启动失败: {exc}")
        return

    main_exit = _process_returncode(self.main_process)
    if self.main_process is None:
      try:
        self._start_main()
      except Exception as exc:
        self.main_process = None
        self.main_started_at = None
        self._backoff(f"自动实盘启动失败: {exc}")
      return

    if main_exit is not None:
      failed_pid = getattr(self.main_process, "pid", "?")
      self.main_process = None
      self.main_started_at = None
      self._backoff(f"自动实盘异常退出 (PID={failed_pid}, 状态={main_exit})")
      if self._emergency_halt_requested():
        print("重启等待期间收到紧急制动，取消拉起自动实盘", flush=True)
        self.request_stop()
      if (not self.stop_requested and not self._emergency_halt_requested()
          and _is_running(self.qmt_process)):
        try:
          self._start_main()
        except Exception as exc:
          self.main_process = None
          self.main_started_at = None
          print(f"自动实盘重启失败: {exc}", flush=True)
      return

    self._reset_backoff_if_stable()

  def shutdown(self):
    self.stop_requested = True
    if self._stop_main("watchdog 正在退出"):
      self._stop_owned_qmt()
    elif _is_running(self.qmt_process):
      print("旧主进程仍存活，保留其 QMT，避免留下断开连接的交易进程", flush=True)

  def run_forever(self):
    print("watchdog 进入 24x7 守护；日内阶段与跨日由 scheduler 管理", flush=True)
    try:
      while not self.stop_requested:
        try:
          self.run_once()
        except Exception as exc:
          # 单轮意外不应让守护进程退出；保留健康资源后低频重试。
          self._backoff(f"守护检查异常: {exc}")
        if not self.stop_requested:
          self.sleep(self.poll_interval)
    finally:
      self.shutdown()


def main(argv=None) -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--individual-config",
    type=str,
    default=DEFAULT_INDIVIDUAL_CONFIG,
    help="Individual_config JSON 文件路径",
  )
  parser.add_argument('--trading-backend', choices=('bridge', 'native'), default='bridge')
  args = parser.parse_args(argv)

  try:
    instance_lock = _acquire_watchdog_lock()
  except WatchdogAlreadyRunning as exc:
    print(str(exc), flush=True)
    return 2

  try:
    # 只有取得 watchdog 单实例锁后才可消费 halt；这样重复调用旧诊断入口不会
    # 抢先清掉仍在生效的紧急制动标记。
    _clear_halt_marker()
  except OSError as exc:
    print(f"无法清除紧急制动标记，拒绝启动: {exc}", flush=True)
    instance_lock.release()
    return 3

  watchdog = TradingWatchdog(args.individual_config, trading_backend=args.trading_backend)

  def handle_stop(signum, frame):
    del frame
    print(f"收到停止信号 {signum}，准备退出 watchdog", flush=True)
    watchdog.request_stop()

  signal.signal(signal.SIGTERM, handle_stop)
  signal.signal(signal.SIGINT, handle_stop)

  print(f"QMT Watchdog 启动 (PID={os.getpid()})", flush=True)
  print(f"使用配置文件: {watchdog.individual_config}", flush=True)
  try:
    watchdog.run_forever()
  finally:
    instance_lock.release()
  return 0


if __name__ == "__main__":
  sys.exit(main())
