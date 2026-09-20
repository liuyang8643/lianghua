"""Download a sealed official SW index research snapshot; no production data writes."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import time
from urllib.request import Request, urlopen
from urllib.parse import urlencode

import numpy as np
import pandas as pd

BASE = 'https://www.swsresearch.com/institute-sw/api/index_publish/'


def download(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    raw = output / 'raw'
    raw.mkdir(exist_ok=True)
    requests_log = []

    def fetch(endpoint: str, params: dict, filename: str) -> dict:
        url = BASE + endpoint + '/?' + urlencode(params)
        target = raw / filename
        if not target.exists():
            req = Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.swsresearch.com/'})
            with urlopen(req, timeout=45) as response:
                body = response.read()
            payload = json.loads(body)
            if str(payload['code']) != '200' or not payload['data']:
                raise ValueError(f'Invalid source response: {url}')
            temporary = target.with_suffix('.tmp')
            temporary.write_bytes(body)
            temporary.replace(target)
            time.sleep(0.3)
        body = target.read_bytes()
        payload = json.loads(body)
        if str(payload['code']) != '200' or not payload['data']:
            raise ValueError(f'Invalid cached source response: {url}')
        requests_log.append({'url': url, 'file': str(target), 'sha256': hashlib.sha256(body).hexdigest(), 'raw_file_mtime_utc':datetime.fromtimestamp(target.stat().st_mtime,timezone.utc).isoformat()})
        return payload

    listing = fetch('current', {'page': 1, 'page_size': 50, 'indextype': '一级行业'}, 'directory.json')['data']
    if len(listing['results']) != listing['count'] or listing['count'] != 31:
        raise ValueError('Expected the video current 31-industry universe')
    frames, coverage = [], []
    for row in listing['results']:
        code = row['swindexcode']
        data = fetch('trend', {'swindexcode': code, 'period': 'DAY'}, f'{code}.json')['data']
        frame = pd.DataFrame(data).rename(columns={'swindexcode':'code','bargaindate':'date','openindex':'open','closeindex':'close','maxindex':'high','minindex':'low'})
        frame = frame[['code','date','open','close','high','low']].copy()
        frame['date'] = pd.to_datetime(frame['date'])
        frame['code'] = frame['code'].astype(str)
        for col in ['open','close','high','low']:
            frame[col] = pd.to_numeric(frame[col], errors='raise')
        if frame.duplicated(['date','code']).any() or not frame['code'].eq(code).all():
            raise ValueError(f'Duplicate dates or wrong code: {code}')
        if not np.isfinite(frame[['open','close','high','low']]).all().all() or (frame[['open','close','high','low']] <= 0).any().any():
            raise ValueError(f'Invalid prices: {code}')
        frame = frame.sort_values('date')
        coverage.append({'code':code,'name':row['swindexname'],'rows':len(frame),'start':str(frame.date.min().date()),'end':str(frame.date.max().date())})
        frames.append(frame)
        print(json.dumps(coverage[-1],ensure_ascii=False),flush=True)
    panel = pd.concat(frames, ignore_index=True).sort_values(['date','code'])
    target = output / 'indices.parquet'
    temp = target.with_suffix('.tmp.parquet')
    panel.to_parquet(temp,index=False)
    pd.testing.assert_frame_equal(panel.reset_index(drop=True),pd.read_parquet(temp).reset_index(drop=True))
    temp.replace(target)
    manifest = {'schema':'sw-official-index-research-v1','sealed_at':datetime.now(timezone.utc).isoformat(),'universe':'current 31 SW L1 indices, matching video stated count; historical published universe not certified','pit_certified':False,'index_theoretical_only':True,'coverage':coverage,'requests':requests_log,'parquet_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')


def download_calendar(output: Path) -> None:
    from xtquant import xtdata
    xtdata.enable_hello=False
    stamps=xtdata.get_trading_dates('SH','19991201','20260911')
    dates=pd.to_datetime(stamps,unit='ms',utc=True).tz_convert('Asia/Shanghai').tz_localize(None).to_numpy(dtype='datetime64[D]')
    if len(dates)<6000 or dates[-1]!=np.datetime64('2026-09-11'):
        raise ValueError('QMT calendar coverage insufficient')
    output.mkdir(parents=True,exist_ok=True)
    np.savez(output/'calendar.npz',trade_dates=dates)
    (output/'calendar_manifest.json').write_text(json.dumps({'source':'QMT public get_trading_dates SH','start':'19991201','end':'20260911','raw_epoch_ms':stamps,'sha256':hashlib.sha256((output/'calendar.npz').read_bytes()).hexdigest()},indent=2),encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--calendar-only',action='store_true')
    args=parser.parse_args()
    (download_calendar if args.calendar_only else download)(args.output)
