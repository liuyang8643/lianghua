from pathlib import Path

import pytest

from trading.watchdog import (
    INITIAL_RESTART_DELAY_SEC,
    TradingWatchdog,
    WatchdogAlreadyRunning,
    _acquire_watchdog_lock,
    _clear_halt_marker,
    _process_returncode,
)


class FakePopen:
    def __init__(self, pid: int, returncode=None):
        self.pid = pid
        self.returncode = returncode

    def poll(self):
        return self.returncode


class FakePsutilProcess:
    def __init__(self, pid: int, running=True, status='running'):
        self.pid = pid
        self.running = running
        self._status = status

    def is_running(self):
        return self.running

    def status(self):
        return self._status


def _watchdog(existing_qmt, mains, *, started_qmt=None, halt_path=None):
    terminated = []
    sleeps = []
    qmt_starts = []

    def start_qmt():
        qmt_starts.append(True)
        return started_qmt

    def popen(command, cwd):
        assert command[1:5] == ['-X', 'utf8', '-m', 'trading.main']
        assert Path(cwd).name == 'WBR-live'
        return mains.pop(0)

    def terminate(process):
        terminated.append(process)
        if hasattr(process, 'returncode'):
            process.returncode = 0
        elif hasattr(process, 'running'):
            process.running = False

    watchdog = TradingWatchdog(
        'configs/config.json',
        qmt_finder=lambda: existing_qmt,
        qmt_starter=start_qmt,
        popen_factory=popen,
        terminate_func=terminate,
        sleep_func=sleeps.append,
        monotonic_func=lambda: 100.0,
        poll_interval=0,
        qmt_startup_wait=5,
        halt_path=str(halt_path or Path.cwd() / '.watchdog-test-halt-missing'),
    )
    return watchdog, terminated, sleeps, qmt_starts


def test_process_state_supports_popen_and_psutil_process():
    popen = FakePopen(11)
    process = FakePsutilProcess(12)
    assert _process_returncode(popen) is None
    assert _process_returncode(process) is None

    popen.returncode = 7
    process.running = False
    assert _process_returncode(popen) == 7
    assert _process_returncode(process) == -1


def test_existing_qmt_is_attached_but_never_terminated():
    existing_qmt = FakePsutilProcess(100)
    main = FakePopen(200)
    watchdog, terminated, sleeps, qmt_starts = _watchdog(existing_qmt, [main])

    watchdog.run_once()
    assert watchdog.qmt_process is existing_qmt
    assert watchdog.qmt_owned is False
    assert watchdog.main_process is main
    assert qmt_starts == []
    assert sleeps == []

    watchdog.shutdown()
    assert terminated == [main]
    assert existing_qmt.running is True


def test_qmt_created_by_watchdog_is_terminated_on_shutdown():
    started_qmt = FakePopen(101)
    main = FakePopen(201)
    watchdog, terminated, sleeps, qmt_starts = _watchdog(
        None, [main], started_qmt=started_qmt)

    watchdog.run_once()
    assert watchdog.qmt_owned is True
    assert qmt_starts == [True]
    assert sleeps == [5]

    watchdog.shutdown()
    assert terminated == [main, started_qmt]


def test_main_crash_restarts_only_main_and_keeps_existing_qmt():
    existing_qmt = FakePsutilProcess(102)
    first_main = FakePopen(202)
    second_main = FakePopen(203)
    watchdog, terminated, sleeps, _ = _watchdog(
        existing_qmt, [first_main, second_main])

    watchdog.run_once()
    first_main.returncode = 1
    watchdog.run_once()

    assert watchdog.main_process is second_main
    assert watchdog.qmt_process is existing_qmt
    assert watchdog.qmt_owned is False
    assert terminated == []
    assert sleeps == [INITIAL_RESTART_DELAY_SEC]
    assert watchdog.restart_delay == INITIAL_RESTART_DELAY_SEC * 2


def test_silent_termination_failure_keeps_old_main_and_blocks_restart():
    dead_qmt = FakePsutilProcess(106, running=False)
    old_main = FakePopen(206)
    replacement = FakePopen(207)
    starts = []
    sleeps = []
    watchdog = TradingWatchdog(
        'configs/config.json',
        qmt_finder=lambda: None,
        qmt_starter=lambda: starts.append('qmt'),
        popen_factory=lambda command, cwd: starts.append('main') or replacement,
        terminate_func=lambda process: None,  # 模拟 terminate_process_tree 吞异常返回。
        sleep_func=sleeps.append,
        halt_path=str(Path.cwd() / '.watchdog-test-halt-missing'),
    )
    watchdog.qmt_process = dead_qmt
    watchdog.main_process = old_main

    watchdog.run_once()

    assert watchdog.main_process is old_main
    assert watchdog.qmt_process is dead_qmt
    assert starts == []
    assert sleeps == [INITIAL_RESTART_DELAY_SEC]


def test_emergency_halt_prevents_main_restart_and_preserves_existing_qmt(tmp_path):
    halt_path = tmp_path / 'trading_watchdog.halt'
    halt_path.write_text('emergency_stop_pid=204\n', encoding='utf-8')
    existing_qmt = FakePsutilProcess(104)
    crashed_main = FakePopen(204, returncode=1)
    watchdog, terminated, sleeps, _ = _watchdog(
        existing_qmt, [FakePopen(205)], halt_path=halt_path)
    watchdog.qmt_process = existing_qmt
    watchdog.main_process = crashed_main

    watchdog.run_once()

    assert watchdog.stop_requested is True
    assert watchdog.main_process is crashed_main
    assert terminated == []
    assert sleeps == []

    watchdog.shutdown()
    assert terminated == []  # 已退出的 main 与既有 QMT 都不得终止。


def test_manual_watchdog_start_clears_previous_halt_marker(tmp_path):
    halt_path = tmp_path / 'trading_watchdog.halt'
    halt_path.write_text('emergency_stop_pid=206\n', encoding='utf-8')

    _clear_halt_marker(str(halt_path))

    assert not halt_path.exists()


def test_watchdog_lock_rejects_second_instance(tmp_path):
    lock_path = tmp_path / 'watchdog.lock'
    first = _acquire_watchdog_lock(str(lock_path))
    try:
        with pytest.raises(WatchdogAlreadyRunning, match='已有一个实盘 watchdog'):
            _acquire_watchdog_lock(str(lock_path))
    finally:
        first.release()
