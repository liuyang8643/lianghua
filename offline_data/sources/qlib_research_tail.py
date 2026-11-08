"""Fetch a bounded, independently stored A-share research extension.

Raw responses are retained. This command never modifies the production store
or synthesizes prices, financial announcements, or missing trading dates.
"""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import socket
import time


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


FIELDS = 'date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,isST'


def validate_rows(rows, code, start, end):
    if not isinstance(rows,list) or not rows:
        raise ValueError('empty or malformed source response')
    previous=None
    for row in rows:
        if not isinstance(row,dict) or set(FIELDS.split(','))-set(row):
            raise ValueError('cached/source row has missing required fields')
        day=date.fromisoformat(row['date']).isoformat()
        if row['code']!=code or not start<=day<=end or (previous is not None and day<=previous):
            raise ValueError('cached/source response has wrong code or date range/order')
        if row['tradestatus'] not in ('0','1') or row['isST'] not in ('0','1'):
            raise ValueError('unknown source trading/ST state')
        previous=day


def fetch(output: Path, start: str, end: str) -> None:
    import baostock as bs
    import numpy as np

    output.mkdir(parents=True, exist_ok=True)
    daily = output / 'daily'
    daily.mkdir(exist_ok=True)
    request={'version':1,'source':'baostock','start':start,'end':end,
             'adjustflag':'3','fields':FIELDS}
    if date.fromisoformat(start)>date.fromisoformat(end):
        raise ValueError('invalid request range')
    request_path=output/'request.json'
    if request_path.exists():
        if json.loads(request_path.read_text(encoding='utf-8'))!=request:
            raise ValueError('cache request identity mismatch')
    else:
        if (output/'stock_basic.json').exists() or any(daily.glob('*.json')):
            raise ValueError('unbound pre-existing cache is not reusable')
        atomic_json(request_path,request)
    socket.setdefaulttimeout(20)
    begun = time.monotonic()
    login = bs.login()
    if login.error_code != '0':
        raise RuntimeError(f'baostock login: {login.error_code} {login.error_msg}')

    def query(result):
        import pandas as pd
        if result.error_code != '0':
            raise RuntimeError(f'baostock: {result.error_code} {result.error_msg}')
        # baostock.get_data uses the removed DataFrame.append on multi-page
        # responses. Consume its public row iterator instead of monkeypatching
        # pandas or silently retaining only the first page.
        rows = []
        while result.next():
            rows.append(result.get_row_data())
        data = pd.DataFrame(rows, columns=result.fields)
        if result.error_code != '0':
            raise RuntimeError(f'baostock stream: {result.error_code} {result.error_msg}')
        return data

    try:
        basic_path = output / 'stock_basic.json'
        if basic_path.exists():
            basic = json.loads(basic_path.read_text(encoding='utf-8'))
        else:
            basic = query(bs.query_stock_basic()).to_dict('records')
            atomic_json(basic_path, basic)
        # Include delisted members that were still listed during the extension.
        selected = [r for r in basic if r.get('type') == '1'
                    and (r['code'].startswith('sh.6') or r['code'].startswith(('sz.0', 'sz.3')))
                    and r.get('ipoDate', '') <= end
                    and (not r.get('outDate') or r['outDate'] >= start)]
        if not selected or len({r['code'] for r in selected})!=len(selected):
            raise ValueError('empty or duplicate eligible source universe')
        atomic_json(output / 'universe.json', selected)
        fields = FIELDS
        failures = []
        rows_total = 0
        for index, stock in enumerate(selected):
            code = stock['code']
            path = daily / (code + '.json')
            if path.exists():
                rows = json.loads(path.read_text(encoding='utf-8'))
                validate_rows(rows,code,start,end)
            else:
                rows = None
                for attempt in range(2):
                    try:
                        frame = query(bs.query_history_k_data_plus(
                            code, fields, start_date=start, end_date=end,
                            frequency='d', adjustflag='3'))
                        rows = frame.to_dict('records')
                        validate_rows(rows,code,start,end)
                        atomic_json(path, rows)
                        break
                    except Exception as exc:
                        if attempt == 1:
                            failures.append({'code': code, 'error': repr(exc)})
                        else:
                            time.sleep(0.5)
                time.sleep(0.04)
            rows_total += len(rows or [])
            if (index + 1) % 50 == 0 or index + 1 == len(selected):
                state = {'state': 'fetching', 'pid': os.getpid(), 'completed': index + 1,
                         'total': len(selected), 'rows': rows_total, 'failures': failures,
                         'elapsed_seconds': time.monotonic() - begun, 'start': start, 'end': end}
                atomic_json(output / 'status.json', state)
                print(json.dumps({k: v for k, v in state.items() if k != 'failures'}), flush=True)
            if (output / 'STOP').exists():
                raise RuntimeError('explicit STOP file')
        state['state'] = 'complete' if not failures else 'incomplete'
        atomic_json(output / 'status.json', state)
    finally:
        bs.logout()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', default='2026-08-20')
    parser.add_argument('--end', default='2026-10-31')
    args = parser.parse_args()
    fetch(args.output, args.start, args.end)
