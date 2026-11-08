"""Verify written Nov 2 market rows against the archived independent source."""
from datetime import date
import hashlib
import json
from pathlib import Path
import pandas as pd
from data.update_tencent_close import parse_close_packet

root = Path(__file__).resolve().parents[2]
cache_path = root / 'data/market_status/tencent_close_2026-11-02.json'
manifest_path = root / 'data/market_status/tencent_close_validation_2026-11-02.json'
cache = json.loads(cache_path.read_text(encoding='utf-8'))
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
assert manifest['written'] and manifest['date'] == '2026-11-02'
checked, failures = 0, []
for packet in cache['packets']:
    rows = parse_close_packet(packet['text'], set(packet['symbols']), date(2026, 11, 2))
    for symbol, quote in rows.items():
        if quote is None:
            continue
        code = symbol[2:] + ('.SH' if symbol.startswith('sh') else '.SZ')
        frame = pd.read_parquet(root / 'data/k-line' / (code + '.parquet'),
                                filters=[('time', '==', quote['time'])])
        if len(frame) != 1:
            failures.append(code + ': row count')
            continue
        actual = frame.iloc[0]
        tolerance = quote['_volume_tolerance_lots'] if code in manifest['backfilled_histories'] else 0.000001
        if (any(abs(actual[key] - quote[key]) > .011 for key in ('open', 'high', 'low', 'close', 'preClose'))
                or abs(actual['amount'] - quote['amount']) > 1
                or abs(actual['volume'] - quote['volume']) > tolerance):
            failures.append(code + ': differs from archived source')
        checked += 1
report = dict(date='2026-11-02', checked=checked, passed=not failures and checked == manifest['rows'],
              failures=failures, source_sha256=hashlib.sha256(cache_path.read_bytes()).hexdigest(),
              manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(), network_calls=0, qmt_calls=0)
Path(__file__).with_name('closing_rows_verified.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
assert report['passed']
