"""Read-only preparation evidence. This never proves real order permissions."""
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
from trading.planner_watchdog import install_no_qmt_guard
install_no_qmt_guard()
from trading.plan_publisher import load_account_file, previous_baseline
from trading.startup_data import validate_startup_runtime
from utils.stock.time import get_last_trading_day

directory = ROOT / 'data/live_file'
now = datetime.now()
cfg = json.loads((directory / 'config.json').read_text())
identity = json.loads((directory / 'identity.json').read_text())
status = json.loads((directory / 'status.json').read_text())
issues = []
same = lambda a, b: os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))

process = psutil.Process(identity['pid'])
parent = psutil.Process(identity['parent_pid'])
if (process.create_time() != identity['created'] or parent.create_time() != identity['parent_created']
        or process.ppid() != parent.pid or process.cmdline() != identity['cmdline']
        or not same(process.cwd(), ROOT) or not same(identity['root'], ROOT)
        or not same(process.exe(), cfg['broker_python'])
        or not any(same(parent.exe(), image) for image in cfg['client_images'])):
    issues.append('executor_identity_mismatch')
if not cfg['live_enabled'] or not status.get('live_enabled') or status['pid'] != process.pid:
    issues.append('live_mode_not_loaded')
if not 0 <= time.time() - status['updated'] < 15:
    issues.append('executor_heartbeat_stale')
if status['stage'] not in ('waiting_for_local_target', 'preview', 'executing', 'completed', 'pending', 'published'):
    issues.append('executor_stage_' + status['stage'])
for path in (directory / 'stop.request', ROOT / 'data/live_trades/trading_watchdog.halt'):
    if path.exists():
        issues.append('stop_marker_' + path.name)
snapshot, asset, positions = load_account_file(directory / 'account_latest.json', cfg['account'], now.date())
previous = get_last_trading_day(now.date() - timedelta(days=1))
previous_baseline(directory, previous, cfg['account'])
try:
    runtime = str(validate_startup_runtime(previous))
except (OSError, ValueError, KeyError) as error:
    runtime = None
    issues.append(str(error))

planners = []
for candidate in psutil.process_iter(['name']):
    if candidate.info['name'] != 'python.exe':
        continue
    try:
        args = candidate.cmdline()
        if (any(args[i:i + 2] == ['-m', 'trading.planner_watchdog'] for i in range(len(args) - 1))
                and same(candidate.cwd(), ROOT)):
            planners.append(candidate)
    except psutil.NoSuchProcess:
        pass
parent_ids = {p.ppid() for p in planners}
workers = [p for p in planners if p.pid not in parent_ids]
if len(workers) != 1:
    issues.append('local_planner_worker_count_' + str(len(workers)))
modules = ['app_executor', 'execution_engine', 'execution_journal', 'target_protocol', 'bridge_wire', 'qmt_bridge']
sources = {}
for name in modules:
    path = ROOT / 'trading' / (name + '.py')
    sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.stat().st_mtime > process.create_time():
        issues.append('source_modified_after_client_start_' + name)
with sqlite3.connect('file:' + (directory / 'execution.sqlite3').as_posix() + '?mode=ro', uri=True) as db:
    intents = db.execute('select count(*) from order_intents').fetchone()[0]
report = dict(checked_at=now.isoformat(), preparation_ready=not issues, issues=issues,
    real_order_permission_verified=False, account=cfg['account'], executor=identity,
    executor_status=status, planner_workers=[dict(pid=p.pid, created=p.create_time()) for p in workers],
    runtime=runtime, expected_history_end=previous.isoformat(), position_count=len(positions),
    account_captured_at=snapshot['captured_at'], order_intent_count=intents, source_hashes=sources)
output = Path(__file__).with_name('preopen_' + now.date().isoformat() + '.json')
output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(dict(report=str(output), preparation_ready=not issues, issues=issues), ensure_ascii=False))
raise SystemExit(bool(issues))
