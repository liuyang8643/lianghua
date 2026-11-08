"""Validate the client execution identity; stop its strategy through files only."""
import json
from pathlib import Path
import psutil

from trading.bridge_wire import atomic_json


def request_file_stop(root, *, process_factory=psutil.Process):
    root = Path(root).resolve()
    directory = root / 'data/live_file'
    identity = json.loads((directory / 'identity.json').read_text(encoding='utf-8'))
    config_path = directory / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    process = process_factory(identity['pid'])
    parent = process_factory(identity['parent_pid'])
    same = lambda a, b: str(Path(a).resolve()).casefold() == str(Path(b).resolve()).casefold()
    if (not process.is_running() or not parent.is_running()
            or abs(process.create_time() - identity['created']) > 0.1
            or abs(parent.create_time() - identity['parent_created']) > 0.1
            or process.ppid() != parent.pid
            or not same(process.exe(), config['broker_python'])
            or not any(same(parent.exe(), p) for p in config['client_images'])
            or process.cmdline() != identity['cmdline']
            or not same(process.cwd(), root)
            or not same(identity['root'], root)
            or not same(config['root'], root)
            or not same(config['individual_config'], root / 'configs/config.json')
            or not same(identity['config'], config_path)
            or identity['account'] != config['account']):
        raise RuntimeError('Client execution identity is stale or belongs to another repository')
    # No broker process is terminated. The per-order guard sees halt immediately;
    # the strategy loop sees stop on its next tick and preserves the journal.
    atomic_json(root / 'data/live_trades/trading_watchdog.halt',
                {'executor_pid': process.pid, 'reason': 'file_executor_emergency_stop'})
    atomic_json(directory / 'stop.request', {'executor_pid': process.pid})
    return process.pid
