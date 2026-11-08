"""Cutover: stop only verified WBR-live watchdog processes, never QMT."""
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
        if not any(args[i:i + 2] == ['-m', 'trading.watchdog'] for i in range(len(args) - 1)):
            continue
        if not same(process.cwd(), root):
            continue
        index = args.index('--individual-config')
        if not same(args[index + 1], root / 'configs/config.json'):
            continue
        selected.append((process, process.create_time(), args))
    except (psutil.Error, ValueError, IndexError):
        continue
stopped = []
for process, created, args in sorted(selected, key=lambda item: item[0].pid, reverse=True):
    try:
        if abs(process.create_time() - created) > 0.1 or process.cmdline() != args:
            raise RuntimeError('Watchdog identity changed before stop')
        process.terminate()  # No recursive tree termination; Qt processes stay alive.
        process.wait(5)
        stopped.append(process.pid)
    except psutil.NoSuchProcess:
        pass
for pid, created in clients.items():
    if abs(psutil.Process(pid).create_time() - created) > 0.1:
        raise RuntimeError('A preserved QMT client process changed')
result = {'stopped_watchdogs': stopped, 'preserved_client_processes': clients,
          'qmt_processes_terminated': 0}
path = root / 'tmp/file_mode_cutover_processes.json'
path.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result))
