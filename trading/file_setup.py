"""Prepare file execution without connecting to or controlling QMT."""
import argparse
from datetime import datetime, time as daytime
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

from trading.account_snapshot import validate_account_snapshot
from trading.bridge_setup import bootstrap_source
from trading.bridge_wire import atomic_json


def prepare(root, *, live=False):
    root = Path(root).resolve()
    source = json.loads((root / 'data/live_bridge/config.json').read_text(encoding='utf-8'))
    directory = root / 'data/live_file'
    directory.mkdir(parents=True, exist_ok=True)
    source.update(execution_mode='file', live_enabled=live, root=str(root),
                  individual_config=str(root / 'configs/config.json'))
    atomic_json(directory / 'config.json', source)
    # Reuse the registered strategy entry; do not restart or edit the running Qt client.
    entry = Path(source['broker_python']).parent.parent / 'mpython/WBR_EXEC_BRIDGE.py'
    backup = directory / 'WBR_EXEC_BRIDGE.before_file_mode.py'
    if entry.exists() and not backup.exists():
        backup.write_bytes(entry.read_bytes())
    entry.write_text(bootstrap_source(root, 'trading/app_executor.py'), encoding='utf-8')
    return directory


def migrate_verified_baseline(root, previous):
    """Recover a real post-close account pair from dated, account-bound RPC facts.

    The original receipt timestamps/IDs are retained. No old file is relabelled
    as a current query, and an unbound Parquet snapshot is never assumed valid.
    """
    root = Path(root).resolve()
    directory = root / 'data/live_file'
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    output = directory / ('account_eod_' + previous.isoformat() + '.json')
    if output.exists():
        return output
    begin = datetime.combine(previous, daytime(15, 35)).timestamp()
    end = datetime.combine(previous, daytime(23, 59, 59)).timestamp()
    database = sqlite3.connect('file:' + (root / 'data/live_bridge/bridge.sqlite3').as_posix() + '?mode=ro', uri=True)
    try:
        rows = database.execute("SELECT id,created,instance,operation,payload,response FROM requests "
                                "WHERE state='done' AND created>=? AND created<=? "
                                "AND operation IN ('asset','positions') ORDER BY created DESC",
                                (begin, end)).fetchall()
    finally:
        database.close()
    accepted = []
    for request_id, created, instance, operation, payload, response in rows:
        request, reply = json.loads(payload), json.loads(response)
        if request['account_id'] == config['account'] and reply['ok']:
            accepted.append((request_id, created, instance, operation, reply['result']))
    for asset in [r for r in accepted if r[3] == 'asset']:
        positions = [r for r in accepted if r[3] == 'positions' and r[2] == asset[2]
                     and abs(r[1] - asset[1]) <= 10]
        if not positions:
            continue
        pair = min(positions, key=lambda r: abs(r[1] - asset[1]))
        if (asset[4].get('account_id') != config['account']
                or any(p.get('account_id') != config['account'] for p in pair[4])):
            continue
        validate_account_snapshot(SimpleNamespace(**asset[4]), [SimpleNamespace(**p) for p in pair[4]])
        snapshot = dict(version=1, account=config['account'], trade_date=previous.isoformat(),
                        captured_at=max(asset[1], pair[1]), asset=asset[4], positions=pair[4],
                        source='verified_legacy_post_close_rpc_receipts',
                        source_requests=[asset[0], pair[0]], source_instance=asset[2])
        atomic_json(output, snapshot)
        return output
    raise RuntimeError('No complete account-bound post-close broker receipt pair is available')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--migrate-eod')
    args = parser.parse_args()
    print(prepare(args.root, live=args.live))
    if args.migrate_eod:
        print(migrate_verified_baseline(args.root, datetime.strptime(args.migrate_eod, '%Y-%m-%d').date()))
