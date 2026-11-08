"""Read-only diagnostic. Never invokes reconcile_file(execute=True)."""
import json
from datetime import datetime
from pathlib import Path

from configs.env import TRADE_ACCOUNT
from trading.bridge_wire import atomic_json, open_database
from trading.qmt_bridge import BrokerBridge
from trading.target_file import reconcile_file
from trading.trader import Trader

root = Path(__file__).resolve().parents[2]
config = json.loads((root / 'data/live_bridge/config.json').read_text())
assert config['live_enabled'] is False
trader = Trader(TRADE_ACCOUNT, backend='native', enable_callbacks=False, enable_orders=False)
target_path = root / 'data/live_bridge/targets/read_only_acceptance.json'
target_path.parent.mkdir(parents=True, exist_ok=True)
now = datetime.now()
target = {'account': config['account'], 'trade_date': now.date().isoformat(),
          'expires': now.timestamp() + 600, 'batch_id': '20261102abcd',
          'targets': {'688701.SH': {'shares': 5911, 'price': 6.74},
                      '688528.SH': {'shares': 4797, 'price': 8.23},
                      '688021.SH': {'shares': 2399, 'price': 16.4},
                      '301049.SZ': {'shares': 2400, 'price': 16.41},
                      '603208.SH': {'shares': 0, 'price': 9.8}}}
atomic_json(target_path, target)
db = open_database(root / 'tmp/qmt_bridge_20261102/readonly_preview.sqlite3')
try:
    bridge = BrokerBridge(config, trader.client, trader.account, db)
    result = reconcile_file(bridge, str(target_path), 0, execute=False)
    result['verified_at'] = now.isoformat()
    result['environment'] = 'local Python read-only native adapter, NOT app Python acceptance'
    atomic_json(root / 'tmp/qmt_bridge_20261102/target_diff_readonly_evidence.json', result)
    print(json.dumps({'dry_run': result['dry_run'], 'diff': result['diff']}, ensure_ascii=False))
finally:
    db.close()
    trader.client.stop()
