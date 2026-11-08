"""Strict announcement-date replay for legacy factor financial inputs.

Uses archived PershareIndex vintages and the same conflict/quarantine
normalizer as production. No statutory disclosure-date approximation or
latest-snapshot backfill. Source fields are documented by the QMT dictionary.
"""
from __future__ import annotations

import hashlib
import inspect
from io import BytesIO
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from offline_data.financial_versions import normalize_financial_events


PERSHARE_FIELDS={
    'eps':'s_fa_eps_basic', 'bps':'s_fa_bps', 'roe':'net_roe',
    'profit_yoy':'du_profit_rate', 'revenue_yoy':'inc_revenue_rate',
    'operating_cf_ps':'s_fa_ocfps', 'gross_margin':'sales_gross_profit',
}
FIELD_NOTES={
    'revenue_yoy':'QMT reported main-business revenue YoY, percentage points; differs from legacy total-operating-revenue YoY',
    'profit_yoy':'QMT reported net-profit YoY, percentage points',
    'roe':'QMT diluted ROE, percentage points',
    'gross_margin':'QMT sales gross margin, percentage points',
    'eps':'QMT reported basic EPS in yuan; not TTM',
    'bps':'QMT reported book value per share in yuan',
    'operating_cf_ps':'QMT reported operating cash flow per share in yuan; not TTM',
}
SOURCE_DOCUMENTATION='https://dict.thinktrader.net/nativeApi/xtdata.html'


def load_pershare_events(directory: Path, stock_codes):
    directory=Path(directory);request=json.loads((directory/'request.json').read_text(encoding='utf-8'))
    if list(stock_codes)[:len(request['codes'])]!=request['codes']:
        raise ValueError('financial archive stock axis is not a prefix of research axis')
    parts=[];hashes={}
    for offset in range(0,len(request['codes']),20):
        batch=directory/f'batch_{offset:05d}'
        manifest=json.loads((batch/'manifest.json').read_text(encoding='utf-8'))
        payload=(batch/'PershareIndex.parquet').read_bytes()
        digest=hashlib.sha256(payload).hexdigest()
        if digest!=manifest['sha256']['PershareIndex']:
            raise ValueError(f'financial archive bytes mismatch: {batch}')
        hashes[str(offset)]=digest
        frame=pd.read_parquet(BytesIO(payload))
        if len(frame):parts.append(frame[['stock_code','m_timetag','m_anntime',*PERSHARE_FIELDS.values()]])
    frame=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=['stock_code','m_timetag','m_anntime',*PERSHARE_FIELDS.values()])
    event=normalize_financial_events(frame,tuple(stock_codes),tuple(PERSHARE_FIELDS.values()))
    identity={'request':request,'table_sha256':hashes,'field_mapping':PERSHARE_FIELDS,'field_notes':FIELD_NOTES,
              'source_documentation':SOURCE_DOCUMENTATION,'normalization_audit':event.audit}
    return event,identity


def replay_latest_fields(dates,n_stocks,event):
    """Stream latest report values, serially applying strictly earlier disclosures."""
    state=np.full((n_stocks,len(event.fields)),np.nan)
    latest=np.full(n_stocks,-1,dtype=np.int64);cursor=0
    for day in np.asarray(dates,dtype='datetime64[D]'):
        stop=int(np.searchsorted(event.announcement_dates,day,side='left'))
        for i in range(cursor,stop):
            col=int(event.stock_columns[i]);quarter=int(event.quarters[i])
            if quarter>=latest[col]:
                state[col]=event.values[i];latest[col]=quarter
        cursor=stop
        yield state.copy()


def materialize_pershare(directory,stock_codes,dates,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    events,identity=load_pershare_events(Path(directory),stock_codes)
    identity.update(dates_sha256=hashlib.sha256(np.asarray(dates,dtype='datetime64[D]').tobytes()).hexdigest(),
                    stocks_sha256=hashlib.sha256(json.dumps(list(stock_codes)).encode()).hexdigest(),
                    implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    normalizer_sha256=hashlib.sha256(inspect.getsource(inspect.getmodule(normalize_financial_events)).encode()).hexdigest())
    signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    receipt=output/'identity.json'
    if receipt.exists():
        prior=json.loads(receipt.read_text(encoding='utf-8'))
        if prior['signature']!=signature:raise ValueError('PIT supplemental financial cache identity mismatch')
        for name,sha in prior['array_sha256'].items():
            with (output/(name+'.npy')).open('rb') as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=sha:raise ValueError('PIT financial cache corrupt')
        return {name:np.load(output/(name+'.npy'),mmap_mode='r') for name in PERSHARE_FIELDS}
    maps={name:np.lib.format.open_memmap(output/(name+'.tmp.npy'),mode='w+',dtype=np.float64,shape=(len(dates),len(stock_codes))) for name in PERSHARE_FIELDS}
    for row,values in enumerate(replay_latest_fields(dates,len(stock_codes),events)):
        for col,name in enumerate(PERSHARE_FIELDS):maps[name][row]=values[:,col]
    for name,array in maps.items():
        array.flush();array._mmap.close();os.replace(output/(name+'.tmp.npy'),output/(name+'.npy'))
    hashes={}
    for name in maps:
        with (output/(name+'.npy')).open('rb') as stream:hashes[name]=hashlib.file_digest(stream,'sha256').hexdigest()
    receipt.write_text(json.dumps({'signature':signature,'identity':identity,'array_sha256':hashes},ensure_ascii=False,indent=2),encoding='utf-8')
    return {name:np.load(output/(name+'.npy'),mmap_mode='r') for name in PERSHARE_FIELDS}
