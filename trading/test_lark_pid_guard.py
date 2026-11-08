"""File-only emergency control validates both strategy and broker parent."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from trading.file_control import request_file_stop


def setup(root):
    directory=root/'data/live_file';directory.mkdir(parents=True)
    broker=str(root/'broker/pythonw.exe');client=str(root/'broker/XtMiniQmt.exe')
    config=dict(root=str(root),broker_python=broker,client_images=[client],account='test',
                individual_config=str(root/'configs/config.json'))
    identity=dict(pid=11,parent_pid=10,created=20.,parent_created=10.,cmdline=[broker,'WBR_EXEC_BRIDGE.py'],
                  root=str(root),config=str(directory/'config.json'),account='test')
    child=NS(pid=11,is_running=lambda:True,create_time=lambda:20.,ppid=lambda:10,
             exe=lambda:broker,cmdline=lambda:identity['cmdline'],cwd=lambda:str(root))
    parent=NS(pid=10,is_running=lambda:True,create_time=lambda:10.,exe=lambda:client)
    (directory/'config.json').write_text(json.dumps(config))
    (directory/'identity.json').write_text(json.dumps(identity))
    return directory,child,parent


@pytest.mark.parametrize('fault',['reused_pid','wrong_parent','foreign_cwd','wrong_exe','changed_command','stale_parent'])
def test_unverified_identity_never_writes_stop_files(tmp_path,fault):
    directory,child,parent=setup(tmp_path)
    if fault=='reused_pid':child.create_time=lambda:22.
    if fault=='wrong_parent':child.ppid=lambda:99
    if fault=='foreign_cwd':child.cwd=lambda:str(tmp_path/'other')
    if fault=='wrong_exe':child.exe=lambda:str(tmp_path/'other.exe')
    if fault=='changed_command':child.cmdline=lambda:['other']
    if fault=='stale_parent':parent.create_time=lambda:9.
    with pytest.raises(RuntimeError,match='identity'):
        request_file_stop(tmp_path,process_factory=lambda pid:{11:child,10:parent}[pid])
    assert not (directory/'stop.request').exists()
    assert not (tmp_path/'data/live_trades/trading_watchdog.halt').exists()


def test_verified_stop_only_writes_markers_and_preserves_broker(tmp_path):
    directory,child,parent=setup(tmp_path)
    # Fake processes have no terminate/kill methods: any such call fails.
    assert request_file_stop(tmp_path,process_factory=lambda pid:{11:child,10:parent}[pid])==11
    assert (directory/'stop.request').is_file()
    assert (tmp_path/'data/live_trades/trading_watchdog.halt').is_file()
