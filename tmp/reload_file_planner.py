"""Reload only this checkout's local planner; never terminate broker processes."""
import json
import os
from pathlib import Path
import psutil

root = Path(__file__).resolve().parents[1]
same = lambda a, b: os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))
clients = {p.pid: p.create_time() for p in psutil.process_iter(['name'])
           if p.info['name'] in ('XtMiniQmt.exe', 'miniquote.exe')}
selected = []
for process in psutil.process_iter(['name']):
    if process.info['name'] != 'python.exe':
        continue
    try:
        args = process.cmdline()
        if not any(args[i:i + 2] == ['-m', 'trading.planner_watchdog'] for i in range(len(args) - 1)):
            continue
        if not same(process.cwd(), root):
            continue
        index = args.index('--config')
        if not same(args[index + 1], root / 'configs/config.json'):
            continue
        selected.append((process, process.create_time(), args, process.ppid()))
    except (psutil.Error, ValueError, IndexError):
        continue
ids = {entry[0].pid for entry in selected}
stopped = []
for process, created, args, parent in sorted(selected, key=lambda row: row[3] in ids, reverse=True):
    try:
        if abs(process.create_time() - created) > 0.1 or process.cmdline() != args:
            raise RuntimeError('Planner identity changed')
        process.terminate()
        process.wait(5)
        stopped.append(process.pid)
    except psutil.NoSuchProcess:
        pass
for pid, created in clients.items():
    if abs(psutil.Process(pid).create_time() - created) > 0.1:
        raise RuntimeError('Broker process unexpectedly changed')
result = dict(stopped_local_planners=stopped, preserved_broker_processes=clients,
              broker_processes_terminated=0)
(root / 'tmp/file_mode_acceptance/planner_reload.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
