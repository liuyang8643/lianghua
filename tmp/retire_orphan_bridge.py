"""Retire the verified old orphan strategy, preserving logged-in Qt processes."""
import ast
import json
import os
from pathlib import Path
import psutil

root = Path(__file__).resolve().parents[1]
cfg = json.loads((root / 'data/live_bridge/config.json').read_text())
heartbeat = json.loads((root / 'data/live_bridge/heartbeat.json').read_text())
backup = ast.parse((root / 'data/live_file/WBR_EXEC_BRIDGE.before_file_mode.py').read_text())
bound_root = next(ast.literal_eval(n.value) for n in backup.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'ROOT' for t in n.targets))
same = lambda a, b: os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))
p = psutil.Process(6500)
args = p.cmdline()
broker = Path(cfg['broker_python'])
assert heartbeat['pid'] == p.pid and heartbeat['connected'] is False
assert abs(p.create_time() - 1793632874.622167) < .01
assert p.ppid() == 22304 and not psutil.pid_exists(22304)
assert same(bound_root, root) and same(cfg['root'], root)
assert same(p.exe(), broker) and same(p.cwd(), broker.parent)
assert same(args[2], broker.parent.parent / 'mpython/WBR_EXEC_BRIDGE.py')
assert same(args[3], cfg['userdata']) and args[4] == '1793632874591'
clients = {pid: psutil.Process(pid).create_time() for pid in (42308, 86120)}
p.terminate()
p.wait(5)
assert all(psutil.Process(pid).create_time() == created for pid, created in clients.items())
result = dict(retired_orphan_pid=6500, created=1793632874.622167,
              missing_former_parent=22304, preserved_qt=clients)
(root / 'tmp/file_mode_acceptance/orphan_retired.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
