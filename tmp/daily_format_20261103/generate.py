from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
from trading.planner_watchdog import install_no_qmt_guard
install_no_qmt_guard()
from trading.lark.sender import lark_sender
# Generation only. Delivery is a separate reviewed step.
def no_send(*args, **kwargs): raise RuntimeError('Generation must not send Feishu')
lark_sender.send_card=no_send
lark_sender.send_file=no_send
lark_sender.update_card=no_send
from trading.file_post_close import reconcile_from_files
from filelock import FileLock
from datetime import date
import json
with FileLock('data/live_file/planner.lock',timeout=0):
 result=reconcile_from_files(date(2026,11,3),'configs/config.json','data/live_file')
 if result is None: raise RuntimeError('Post-close returned no report')
 Path('tmp/daily_format_20261103/report_result.json').write_text(json.dumps(result,ensure_ascii=False,default=str),encoding='utf-8')
 print(json.dumps({'summary':result['summary'],'execution':{k:v for k,v in result['execution'].items() if k not in ('rows','reference_prices')},'reconcile':result['reconcile']},ensure_ascii=False,default=str))
