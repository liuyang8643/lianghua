import json
from datetime import date, datetime
from pathlib import Path

from trading.startup_data import validate_startup_runtime

result = {'checked_at': datetime.now().isoformat(),
          'required_previous_trade_date': '2026-11-02'}
try:
    validate_startup_runtime(date(2026, 11, 2))
    result['runtime_ready'] = True
except Exception as error:
    result['runtime_ready'] = False
    result['error'] = str(error)
Path(__file__).with_name('next_open_data_readiness.json').write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False))
