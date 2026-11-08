from pathlib import Path
import sys
import pytest
from trading.planner_control import stop_planner


class Process:
    def __init__(self,pid,root,module='trading.planner_watchdog'):
        self.pid=pid;self.root=root;self.module=module;self.created=float(pid);self.stopped=False
    def cmdline(self):return [sys.executable,'-m',self.module,'--config',str(self.root/'configs/config.json'),'--directory',str(self.root/'data/live_file')]
    def cwd(self):return str(self.root)
    def exe(self):return sys.executable
    def create_time(self):return self.created
    def is_running(self):return True
    def terminate(self):self.stopped=True
    def wait(self,timeout):return 0


def test_restart_scope_excludes_other_repositories_and_broker_executor(tmp_path):
    current=Process(11,tmp_path)
    other=Process(12,tmp_path/'other')
    client=Process(13,tmp_path,'trading.app_executor')
    data=Process(14,tmp_path,'data.update_all')
    legacy_data=Process(15,tmp_path,'trading.main')
    legacy_data.cmdline=lambda:[sys.executable,'-m','trading.main','--skip','202611061600']
    objects={p.pid:p for p in (current,other,client,data,legacy_data)}
    assert stop_planner(tmp_path,processes=list(objects.values()),process_factory=objects.__getitem__)==[11]
    assert current.stopped and not other.stopped and not client.stopped and not data.stopped
    assert not legacy_data.stopped


def test_reused_pid_is_not_terminated(tmp_path):
    process=Process(11,tmp_path)
    def changed(pid):process.created+=1;return process
    with pytest.raises(RuntimeError,match='identity changed'):
        stop_planner(tmp_path,processes=[process],process_factory=changed)
    assert not process.stopped
