"""Real Windows Job tests with sleeping stdlib children only; never loads training."""
import ctypes as ct
from ctypes import wintypes as wt
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import pytest

if os.name != 'nt':
    pytest.skip('Windows Job Objects', allow_module_level=True)

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/run_research_guarded.py'
spec = importlib.util.spec_from_file_location('research_guard', SOURCE)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def alive(pid):
    api = ct.WinDLL('kernel32', use_last_error=True)
    api.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
    api.OpenProcess.restype = wt.HANDLE
    api.WaitForSingleObject.argtypes = [wt.HANDLE, wt.DWORD]
    api.WaitForSingleObject.restype = wt.DWORD
    api.CloseHandle.argtypes = [wt.HANDLE]
    handle = api.OpenProcess(0x100000, False, pid)
    if not handle:
        return False
    try:
        return api.WaitForSingleObject(handle, 0) == 258
    finally:
        api.CloseHandle(handle)


@unittest.skipUnless(os.name == 'nt', 'Windows Job Objects')
class GuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.options = dict(cwd=self.root, event_path=self.root/'events.jsonl',
            stdout_path=self.root/'stdout.log', stderr_path=self.root/'stderr.log')

    def launch(self, code, detector):
        return guard.run_guarded([sys.executable, '-c', code], detector=detector, **self.options)

    def assert_dead(self, pid):
        deadline = time.monotonic() + 3
        while alive(pid) and time.monotonic() < deadline:
            time.sleep(.05)
        self.assertFalse(alive(pid), f'owned child {pid} survived')

    def test_game_blocks_before_any_child_code(self):
        result = self.launch("from pathlib import Path; Path('executed').touch()",
            lambda: [{'pid': 123, 'image': 'test-only game signal'}])
        self.assertEqual(result, 75)
        self.assertFalse((self.root/'executed').exists())

    def test_worker_proves_membership_in_its_specific_guard_job(self):
        code = f"import sys; sys.path.insert(0, {str(SOURCE.parent)!r}); import run_research_guarded as g; g.require_guarded_job()"
        self.assertEqual(self.launch(code, lambda: []), 0)
        environment = dict(os.environ)
        environment.pop(guard.JOB_ENV, None)
        direct = subprocess.run([sys.executable, '-c', code], env=environment, capture_output=True, timeout=5)
        self.assertNotEqual(direct.returncode, 0)
        self.assertIn(b'Worker must be launched', direct.stderr)

    def test_forged_job_name_without_membership_is_rejected(self):
        api = guard.win_api()
        name = 'Local\\WBRResearchGuard-test-' + str(os.getpid())
        job = guard.checked(api.CreateJobObjectW(None, name))
        try:
            code = f"import sys; sys.path.insert(0, {str(SOURCE.parent)!r}); import run_research_guarded as g; g.require_guarded_job()"
            environment = dict(os.environ)
            environment[guard.JOB_ENV] = name
            direct = subprocess.run([sys.executable, '-c', code], env=environment, capture_output=True, timeout=5)
            self.assertNotEqual(direct.returncode, 0)
            self.assertIn(b'Worker is not owned', direct.stderr)
        finally:
            guard.checked(api.CloseHandle(job))

    def test_game_arriving_during_launch_blocks_suspended_child(self):
        calls = iter([[], [{'pid': 123, 'image': 'test-only game signal'}]])
        result = self.launch("from pathlib import Path; Path('executed').touch()", lambda: next(calls))
        self.assertEqual(result, 75)
        self.assertFalse((self.root/'executed').exists())

    def test_running_game_signal_kills_owned_tree_only(self):
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        self.addCleanup(lambda: (unrelated.kill(), unrelated.wait()) if unrelated.poll() is None else None)
        code = """import subprocess, sys, os, time, json
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
Path('owned.json').write_text(json.dumps([os.getpid(), child.pid]))
time.sleep(30)
"""
        deadline = time.monotonic() + 5
        def detector():
            if (self.root/'owned.json').exists():
                return [{'pid': 123, 'image': 'test-only game signal'}]
            if time.monotonic() > deadline:
                raise TimeoutError('test child failed to start')
            return []
        self.assertEqual(self.launch(code, detector), 75)
        for pid in json.loads((self.root/'owned.json').read_text()):
            self.assert_dead(pid)
        self.assertIsNone(unrelated.poll())

    def test_detector_failure_kills_owned_process(self):
        code = "import os,time; from pathlib import Path; Path('pid').write_text(str(os.getpid())); time.sleep(30)"
        deadline = time.monotonic() + 5
        def detector():
            if (self.root/'pid').exists():
                raise RuntimeError('simulated detection failure')
            if time.monotonic() > deadline:
                raise TimeoutError('test child failed to start')
            return []
        with self.assertRaisesRegex(RuntimeError, 'simulated detection failure'):
            self.launch(code, detector)
        self.assert_dead(int((self.root/'pid').read_text()))

    def test_normal_root_exit_cleans_remaining_child_and_preserves_exit_code(self):
        code = """import subprocess,sys
from pathlib import Path
p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
Path('pid').write_text(str(p.pid))
sys.exit(7)
"""
        self.assertEqual(self.launch(code, lambda: []), 7)
        self.assert_dead(int((self.root/'pid').read_text()))

    def test_guard_abrupt_exit_kills_child_tree(self):
        helper = self.root/'helper.py'
        helper.write_text("""import importlib.util, sys, os
from pathlib import Path
spec=importlib.util.spec_from_file_location('guard',sys.argv[1])
g=importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
def detector():
    if Path('owned.json').exists(): os._exit(91)
    return []
g.run_guarded([sys.executable,'-c',sys.argv[2]],cwd=Path.cwd(),
    event_path='events.jsonl',stdout_path='stdout.log',stderr_path='stderr.log',detector=detector)
""", encoding='utf-8')
        child = """import subprocess,sys,os,json,time
from pathlib import Path
p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
Path('owned.json').write_text(json.dumps([os.getpid(),p.pid]))
time.sleep(30)
"""
        completed = subprocess.run([sys.executable, str(helper), str(SOURCE), child],
            cwd=self.root, timeout=8, capture_output=True)
        self.assertEqual(completed.returncode, 91, completed.stderr)
        for pid in json.loads((self.root/'owned.json').read_text()):
            self.assert_dead(pid)

    def test_crash_immediately_after_atomic_creation_leaves_no_suspended_orphan(self):
        helper = self.root/'creation_crash.py'
        helper.write_text("""import importlib.util,sys,os
from pathlib import Path
spec=importlib.util.spec_from_file_location('guard',sys.argv[1])
g=importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
create=g.create_owned_process
def crash(*args):
    process=create(*args)
    Path('pid').write_text(str(process.dwProcessId))
    os._exit(92)
g.create_owned_process=crash
g.run_guarded([sys.executable,'-c',"from pathlib import Path; Path('executed').touch()"],
    cwd=Path.cwd(),event_path='events.jsonl',stdout_path='stdout.log',stderr_path='stderr.log',detector=lambda: [])
""", encoding='utf-8')
        completed = subprocess.run([sys.executable, str(helper), str(SOURCE)],
            cwd=self.root, timeout=8, capture_output=True)
        self.assertEqual(completed.returncode, 92, completed.stderr)
        self.assert_dead(int((self.root/'pid').read_text()))
        self.assertFalse((self.root/'executed').exists())

    def test_real_game_detection_blocks_harmless_cli(self):
        detected = guard.game_processes()
        if not detected:
            self.skipTest('No real game client open')
        result = subprocess.run([sys.executable, str(SOURCE), '--cwd', str(self.root),
            '--logs', str(self.root/'cli'), '--', sys.executable, '-c',
            "from pathlib import Path; Path('executed').touch()"], capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 75, result.stderr)
        self.assertFalse((self.root/'executed').exists())


if __name__ == '__main__':
    unittest.main()
