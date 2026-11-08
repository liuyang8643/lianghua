"""Run as a NEW read-only client strategy; writes app-Python evidence only."""
import json
import os
from pathlib import Path
import sys
import time

from trading.bridge_wire import atomic_json, open_database
from trading.qmt_bridge import BrokerBridge, process_image
from trading.target_file import reconcile_file


def main():
    from xtquant.xttrader import XtQuantTrader
    from xtquant.xttype import StockAccount
    root = Path(__file__).resolve().parents[1]
    directory = root / 'data/live_bridge'
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    parent = process_image(os.getppid())
    normalize = lambda value: os.path.normcase(os.path.realpath(value))
    if normalize(parent) not in [normalize(path) for path in config['client_images']]:
        raise RuntimeError('This probe must be started inside the brokerage strategy app')
    config['live_enabled'] = False
    client = XtQuantTrader(config['userdata'], int(time.time() * 1000) % 2000000000)
    database = open_database(directory / 'readonly_probe.sqlite3')
    try:
        client.set_timeout(5)
        client.start()
        if client.connect() != 0:
            raise RuntimeError('Read-only probe connection failed')
        account = StockAccount(config['account'], 'STOCK')
        if client.subscribe(account) != 0:
            raise RuntimeError('Read-only probe subscription failed')
        bridge = BrokerBridge(config, client, account, database)
        result = reconcile_file(bridge, str(directory / 'targets/read_only_acceptance.json'),
                                os.getpid(), execute=False)
        result.update(environment='brokerage app Python, read-only probe',
                      python=sys.version, executable=sys.executable, pid=os.getpid(),
                      parent_pid=os.getppid(), parent_image=parent, verified_at=time.time(),
                      actual_order_or_cancel_calls=0)
        atomic_json(directory / 'app_target_diff_evidence.json', result)
        print('WBR_TARGET_DIFF_READ_ONLY_PASS', json.dumps(result, ensure_ascii=True), flush=True)
    finally:
        client.stop()
        database.close()


if __name__ == '__main__':
    main()
