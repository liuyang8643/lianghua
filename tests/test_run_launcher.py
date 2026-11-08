"""Exercise startup with fake processes; never launch a production planner."""
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import run as launcher
from trading.planner_watchdog import apply_start_markers


@pytest.fixture
def setup_launcher(tmp_path, monkeypatch):
    for relative in ('configs/config.json', 'data/live_file/config.json', '.venv/Scripts/python.exe'):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(launcher, 'ROOT', tmp_path)
    sync = Mock()
    process = Mock(pid=123)
    process.wait.side_effect = subprocess.TimeoutExpired('fake', 1)
    popen = Mock(return_value=process)
    monkeypatch.setattr(launcher.subprocess, 'run', sync)
    monkeypatch.setattr(launcher.subprocess, 'Popen', popen)
    monkeypatch.setattr(launcher.subprocess, 'CREATE_NO_WINDOW', 0x08000000, raising=False)
    return tmp_path, sync, popen


def stop_markers(root):
    paths = (root / 'data/live_trades/trading_watchdog.halt', root / 'data/live_file/stop.request')
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('halt', encoding='utf-8')
    return paths


def test_launch_uses_locked_sync_project_python_and_hidden_logs(setup_launcher, monkeypatch, tmp_path):
    root, sync, popen = setup_launcher
    markers = stop_markers(root)
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    launcher.launch()
    sync.assert_called_once_with(['uv', 'sync', '--locked'], cwd=root, check=True)
    args, kwargs = popen.call_args
    assert args[0] == [str(root / '.venv/Scripts/python.exe'), '-u', '-m',
                       'trading.planner_watchdog', '--config', str(root / 'configs/config.json'),
                       '--directory', str(root / 'data/live_file'), '--manual-start']
    assert kwargs['cwd'] == root
    assert kwargs['creationflags'] == 0x08000000
    assert kwargs['env']['PYTHONUTF8'] == '1'
    assert kwargs['env']['PYTHONPATH'] == str(root)
    assert kwargs['stdout'].name.endswith('.out.log')
    assert kwargs['stderr'].name.endswith('.err.log')
    # Only the child, after taking its singleton lock, may clear these.
    assert all(path.exists() for path in markers)


def test_sync_failure_does_not_clear_halt_or_start(setup_launcher):
    root, sync, popen = setup_launcher
    markers = stop_markers(root)
    sync.side_effect = subprocess.CalledProcessError(1, 'uv')
    with pytest.raises(subprocess.CalledProcessError):
        launcher.launch()
    popen.assert_not_called()
    assert all(path.exists() for path in markers)


def test_automated_start_refuses_halt(setup_launcher):
    root, sync, popen = setup_launcher
    markers = stop_markers(root)
    with pytest.raises(RuntimeError, match='automated startup refused'):
        launcher.launch(preserve_halt=True)
    sync.assert_not_called()
    popen.assert_not_called()
    assert all(path.exists() for path in markers)


def test_automated_child_rechecks_halt_created_during_sync(setup_launcher):
    root, sync, popen = setup_launcher
    launcher.launch(preserve_halt=True)
    assert popen.call_args.args[0][-1] == '--preserve-halt'
    markers = stop_markers(root)
    with pytest.raises(RuntimeError, match='automated startup refused'):
        apply_start_markers(root / 'data/live_file', preserve_halt=True)
    assert all(path.exists() for path in markers)


def test_manual_start_clears_only_own_stop_markers(setup_launcher):
    root, _, _ = setup_launcher
    markers = stop_markers(root)
    journal = root / 'data/live_file/execution.json'
    journal.write_text('preserve', encoding='utf-8')
    apply_start_markers(root / 'data/live_file', manual_start=True)
    assert not any(path.exists() for path in markers)
    assert journal.read_text(encoding='utf-8') == 'preserve'


def test_immediate_child_failure_is_not_reported_as_success(setup_launcher):
    _, _, popen = setup_launcher
    popen.return_value.wait.side_effect = None
    popen.return_value.wait.return_value = 1
    with pytest.raises(RuntimeError, match='Planner exited'):
        launcher.launch()


def test_duplicate_planner_cannot_clear_stop_markers(setup_launcher, monkeypatch):
    from filelock import FileLock, Timeout
    from trading.planner_watchdog import run

    root, _, _ = setup_launcher
    markers = stop_markers(root)
    directory = root / 'data/live_file'
    # Stub imports as well as processes: no data or brokerage access in this test.
    monkeypatch.setitem(sys.modules, 'trading.plan_publisher', SimpleNamespace(
        publish_target=Mock(), ensure_local_history=Mock()))
    monkeypatch.setitem(sys.modules, 'utils.stock.time', SimpleNamespace(
        get_last_trading_day=Mock(), _get_trading_calendar_state=Mock()))
    with FileLock(str(directory / 'planner.lock')):
        with pytest.raises(Timeout):
            run(root / 'configs/config.json', directory, manual_start=True)
    assert all(path.exists() for path in markers)
