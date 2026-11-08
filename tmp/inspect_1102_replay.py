import json
import sqlite3
from datetime import datetime
import pandas as pd

db = sqlite3.connect('data/live_bridge/bridge.sqlite3')
rows = db.execute("SELECT id,created,instance,operation,response FROM requests "
                  "WHERE created>=? AND created<? AND state='done' "
                  "AND operation IN ('asset','positions') ORDER BY created",
                  (datetime(2026, 11, 2, 9).timestamp(), datetime(2026, 11, 2, 9, 30).timestamp())).fetchall()
print([(r[0], datetime.fromtimestamp(r[1]).isoformat(), r[2], r[3], json.loads(r[4])['ok']) for r in rows])
print(pd.read_parquet('data/live_trades/plan_2026-11-02.parquet').to_string(index=False))
print(pd.read_parquet('data/live_trades/positions_2026-10-30.parquet')[['code', 'volume']].to_string(index=False))
