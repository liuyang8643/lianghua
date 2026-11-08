import os
import time
from pathlib import Path

from trading.lark.receiver import _arm_watchdog_halt, _resolve_trading_main_process


class FakeProcess:
    def __init__(self, pid, *, command, created_at, running=True,
                 working_dir=None):
        self.pid = pid
        self._command = command
        self._created_at = created_at
        self._running = running
        self._working_dir = working_dir or Path(__file__).resolve().parents[1]

    def is_running(self):
        return self._running

    def cmdline(self):
        return self._command

    def create_time(self):
        return self._created_at

    def cwd(self):
        return str(self._working_dir)


def _pid_file(tmp_path, pid, written_at):
    path = tmp_path / 'trading_main.pid'
    path.write_text(str(pid), encoding='utf-8')
    os.utime(path, (written_at, written_at))
    return path


def test_pid_guard_accepts_matching_trading_main(tmp_path):
    now = time.time()
    pid_path = _pid_file(tmp_path, 321, now)
    expected = FakeProcess(
        321,
        command=['.venv/Scripts/python.exe', '-m', 'trading.main'],
        created_at=now - 2,
    )

    process, reason = _resolve_trading_main_process(
        pid_path, process_factory=lambda pid: expected)

    assert process is expected
    assert reason == ''


def test_pid_guard_rejects_reused_pid_with_unrelated_command(tmp_path):
    now = time.time()
    pid_path = _pid_file(tmp_path, 322, now)
    unrelated = FakeProcess(
        322,
        command=['python.exe', '-m', 'ai.rl.train', '--label', 'trading.main'],
        created_at=now,
    )

    process, reason = _resolve_trading_main_process(
        pid_path, process_factory=lambda pid: unrelated)

    assert process is None
    assert '不是 trading.main' in reason


def test_pid_guard_rejects_stale_file_even_if_pid_now_runs_trading_main(tmp_path):
    now = time.time()
    pid_path = _pid_file(tmp_path, 323, now - 3600)
    reused = FakeProcess(
        323,
        command=['python.exe', '-m', 'trading.main'],
        created_at=now,
    )

    process, reason = _resolve_trading_main_process(
        pid_path, process_factory=lambda pid: reused)

    assert process is None
    assert '疑似陈旧 PID' in reason


def test_pid_guard_rejects_recent_trading_main_from_other_repository(tmp_path):
    now = time.time()
    pid_path = _pid_file(tmp_path, 325, now)
    other_repo = tmp_path / 'other-wbr'
    other = FakeProcess(
        325,
        command=[
            'python.exe', '-m', 'trading.main',
            '--individual-config', str(other_repo / 'configs' / 'config.json'),
        ],
        created_at=now,
        working_dir=other_repo,
    )

    process, reason = _resolve_trading_main_process(
        pid_path, process_factory=lambda pid: other)

    assert process is None
    assert '工作目录不是当前仓库' in reason


def test_pid_guard_rejects_other_config_even_with_current_repo_cwd(tmp_path):
    now = time.time()
    pid_path = _pid_file(tmp_path, 326, now)
    process_with_other_config = FakeProcess(
        326,
        command=[
            'python.exe', '-m', 'trading.main',
            '--individual-config', str(tmp_path / 'config.json'),
        ],
        created_at=now,
    )

    process, reason = _resolve_trading_main_process(
        pid_path, process_factory=lambda pid: process_with_other_config)

    assert process is None
    assert '配置不是当前仓库实盘配置' in reason


def test_pid_guard_rejects_invalid_pid_file(tmp_path):
    pid_path = tmp_path / 'trading_main.pid'
    pid_path.write_text('not-a-pid', encoding='utf-8')

    process, reason = _resolve_trading_main_process(pid_path)

    assert process is None
    assert '内容无效' in reason


def test_emergency_stop_arms_watchdog_halt_before_termination(tmp_path):
    halt_path = tmp_path / 'nested' / 'trading_watchdog.halt'

    _arm_watchdog_halt(324, halt_path)

    assert halt_path.read_text(encoding='utf-8') == 'emergency_stop_pid=324\n'
