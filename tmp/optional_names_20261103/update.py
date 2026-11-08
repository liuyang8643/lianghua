from pathlib import Path
import sys,json
sys.path.insert(0,str(Path.cwd()))
from trading.planner_watchdog import install_no_qmt_guard
install_no_qmt_guard()
from trading.post_close import run_update_all
from trading.startup_data import validate_startup_runtime
from types import SimpleNamespace
from filelock import FileLock
from datetime import date,datetime
with FileLock('data/live_file/planner.lock',timeout=0):
 ok=run_update_all(SimpleNamespace(_skip_update=False),history_days=1,allow_qmt=False)
 if not ok:raise RuntimeError('Required data or Runtime validation failed')
 runtime=validate_startup_runtime(date(2026,11,3))
 proof=dict(checked_at=datetime.now().isoformat(),required_update_passed=True,readiness_for='2026-11-04',runtime=str(runtime),optional_history=json.loads(Path('data/stock_name/name_history_status_2026-11-03.json').read_text(encoding='utf-8')))
 Path('tmp/optional_names_20261103/readiness.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(proof,ensure_ascii=False))
