"""Stop only this turn's local history repair command, preserving account apps."""
import json
import os
from pathlib import Path
import psutil

root = Path(__file__).resolve().parents[1]
same = lambda a, b: os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))
clients = {p.pid: p.create_time() for p in psutil.process_iter(['name'])
           if p.info['name'] in ('XtMiniQmt.exe', 'miniquote.exe', 'pythonw.exe')}
selected = []
for p in psutil.process_iter(['name']):
    if p.info['name'] != 'python.exe':
        continue
    try:
        args = p.cmdline()
        if '-c' in args and 'NOV2_RUNTIME_VERIFIED' in args[-1] and same(p.cwd(), root):
            selected.append((p, p.create_time(), args, p.ppid()))
    except psutil.NoSuchProcess:
        pass
ids = {x[0].pid for x in selected}
for p, created, args, parent in sorted(selected, key=lambda x: x[3] in ids, reverse=True):
    try:
        assert p.create_time() == created and p.cmdline() == args
        p.terminate()
        p.wait(5)
    except psutil.NoSuchProcess:
        pass
assert all(psutil.Process(pid).create_time() == created for pid, created in clients.items())
print(json.dumps(dict(stopped_local_history=list(ids), preserved_clients=list(clients))))
