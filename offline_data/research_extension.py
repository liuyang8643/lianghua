"""Build an immutable research-only extension of the local runtime.

Original price/member/financial cells are preserved. New bars carry source
trading status, and new report vintages activate strictly after announcements.
No production store, live configuration, or old research root is overwritten.
"""
from __future__ import annotations
import argparse
from io import BytesIO
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd

from offline_data.financial_versions import FINANCIAL_FIELDS,FINANCIAL_PANEL_FIELDS,normalize_financial_events,iter_financial_fields
from offline_data.financial_research import PERSHARE_FIELDS,replay_latest_fields,FIELD_NOTES
from offline_data.sources.qlib_research_tail import validate_rows
from utils.atomic_file import atomic_write_json,file_sha256,replace_file


EXTRA_FIELDS=('financial_inventory','financial_other_payable','financial_receivable','financial_operating_cash_flow',
              'financial_cash_taxes_payable','financial_cash_other_payable','financial_cash_other_current_liability','financial_sales_cash')


def read_financial_tables(parent,extra,codes,cutoff):
    """Verify archived batches; do not overwrite old vintages with fresh ones."""
    return read_financial_chain([
        {'directory':str(parent),'known_after':None},
        {'directory':str(extra),'known_after':cutoff,'all_capital_history':True},
    ],codes)


def read_financial_chain(archives,codes):
    """Append announcement vintages from successive frozen acquisition windows."""
    columns={**FINANCIAL_FIELDS,'PershareIndex':tuple(set(FINANCIAL_FIELDS['PershareIndex'])|set(PERSHARE_FIELDS.values())),
             'Capital':('total_capital',)}
    collected={table:[] for table in columns};inputs={}
    old_codes=set()
    for source in archives:
        directory=Path(source['directory']);cutoff=source.get('known_after')
        request=json.loads((directory/'request.json').read_text(encoding='utf-8'))
        if list(codes)[:len(request['codes'])]!=request['codes']:raise ValueError('financial stock axes do not align')
        for offset in range(0,len(request['codes']),20):
            batch=directory/f'batch_{offset:05d}';manifest=json.loads((batch/'manifest.json').read_text(encoding='utf-8'))
            for table,fields in columns.items():
                if table not in request['tables']:continue
                path=batch/(table+'.parquet');payload=path.read_bytes()
                import hashlib
                digest=hashlib.sha256(payload).hexdigest()
                if digest!=manifest['sha256'][table]:raise ValueError('financial batch changed: '+str(path))
                inputs[str(path)]=digest
                frame=pd.read_parquet(BytesIO(payload))
                if frame.empty:continue
                if cutoff is not None and (table!='Capital' or not source.get('all_capital_history',False)):
                    announcement=pd.to_datetime(frame['m_anntime'].astype(str),format='%Y%m%d',errors='coerce')
                    frame=frame[(announcement>pd.Timestamp(cutoff))|~frame['stock_code'].isin(old_codes)]
                collected[table].append(frame[['stock_code','m_timetag','m_anntime',*fields]])
        old_codes.update(request['codes'])
    frames={table:pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=['stock_code','m_timetag','m_anntime',*columns[table]])
            for table,parts in collected.items()}
    return frames,inputs


def build(root,base_path,*,financial_chain=None):
    from scripts.run_research_guarded import require_guarded_job
    require_guarded_job()
    begun=time.monotonic();output=root/'extended_runtime';output.mkdir(exist_ok=False)
    fields_dir=output/'fields';fields_dir.mkdir()
    base=np.load(base_path,allow_pickle=False);old_dates=base['trade_dates'];old_codes=tuple(base['stock_codes'].astype(str))
    with np.load(root/'extended_axes.npz',allow_pickle=False) as axes:codes=tuple(axes['stock_codes'].astype(str))
    if codes[:len(old_codes)]!=old_codes:raise ValueError('extension must preserve original stock order')
    source=root/'tail_source';audit=json.loads((root/'tail_postvalidation.json').read_text(encoding='utf-8'))
    if audit['state']!='validated':raise ValueError('daily extension has not passed source validation')
    tail_dates=np.asarray([d for d in audit['calendar'] if np.datetime64(d)>old_dates[-1]],dtype='datetime64[D]')
    dates=np.r_[old_dates,tail_dates];n0=len(old_dates);s0=len(old_codes);shape=(len(dates),len(codes))
    lookup={code:i for i,code in enumerate(codes)};date_index={str(d):i for i,d in enumerate(dates)}
    basic=json.loads((source/'stock_basic.json').read_text(encoding='utf-8'))
    basic={r['code'].split('.')[1]+'.'+r['code'].split('.')[0].upper():r for r in basic}
    bars=[]
    for symbol,digest in audit['response_sha256'].items():
        path=source/'daily'/(symbol+'.json')
        if file_sha256(path)!=digest:raise ValueError('daily source bytes changed')
        rows=json.loads(path.read_text(encoding='utf-8'));validate_rows(rows,symbol,audit['request']['start'],audit['request']['end'])
        code=symbol.split('.')[1]+'.'+symbol.split('.')[0].upper();col=lookup[code]
        for row in rows:
            i=date_index[row['date']]
            if i>=n0 or col>=s0:bars.append((i,col,row))
    issue={r['stock_code']:r for r in json.loads((root/'new_issue_references.json').read_text(encoding='utf-8')) if 'error' not in r}
    arrays={};inherited_fields=set(base.files)
    for name in base.files:
        if name in ('stock_codes','trade_dates'):continue
        original=base[name];target_shape=shape if original.ndim==2 else (len(codes),)
        array=np.lib.format.open_memmap(fields_dir/(name+'.npy'),mode='w+',dtype=original.dtype,shape=target_shape)
        if array.dtype.kind=='f':array[:]=np.nan
        elif array.dtype.kind=='M':array[:]=np.datetime64('NaT')
        elif name=='listing_age':array[:]=-1
        else:array[:]=False
        if original.ndim==2:array[:n0,:s0]=original
        else:array[:s0]=original
        if name in ('open','high','low','close','preClose','volume','amount'):
            key='preclose' if name=='preClose' else name
            for i,col,row in bars:
                if name!='preClose' and row['tradestatus']!='1':continue
                value=row[key]
                array[i,col]=float(value) if value else np.nan
        elif name=='st_mask':
            array[n0:,:s0]=original[-1]
            for i,col,row in bars:array[i,col]=row['isST']=='1'
        elif name=='delisted_mask':
            array[n0:,:s0]=original[-1]
            for code,col in lookup.items():
                out=basic.get(code,{}).get('outDate','')
                if out:array[n0:,col]|=tail_dates>np.datetime64(out)
        elif name=='listing_age':
            array[n0:,:s0]=np.where(original[-1]>=0,original[-1]+np.arange(1,len(tail_dates)+1)[:,None],-1)
            for code in codes[s0:]:
                at=int(np.searchsorted(dates,np.datetime64(basic[code]['ipoDate'])))
                array[at:,lookup[code]]=np.arange(len(dates)-at)
        elif name=='total_share':array[n0:,:s0]=original[-1]
        elif name=='issue_date':
            for code in codes[s0:]:array[lookup[code]]=np.datetime64(basic[code]['ipoDate'])
        elif name=='issue_price':
            for code in codes[s0:]:
                if code not in issue:raise ValueError('missing IPO reference '+code)
                if issue[code]['list_date']!=basic[code]['ipoDate']:raise ValueError('IPO listing dates disagree '+code)
                array[lookup[code]]=issue[code]['issue_price']
        array.flush();arrays[name]=array;del original
        atomic_write_json(output/'status.json',{'state':'raw_fields','field':name,'elapsed_seconds':time.monotonic()-begun})
    base.close()
    if financial_chain is None:
        base_manifest=json.loads(Path(base_path).with_suffix('.manifest.json').read_text(encoding='utf-8'))
        frames,input_hashes=read_financial_tables(Path(base_manifest['financial_directory']),root/'financial_tail',codes,str(old_dates[-1]))
    else:
        frames,input_hashes=read_financial_chain(financial_chain,codes)
    core={table:normalize_financial_events(frames[table],codes,fields) for table,fields in FINANCIAL_FIELDS.items()}
    for name in EXTRA_FIELDS:
        if name not in arrays:arrays[name]=np.lib.format.open_memmap(fields_dir/(name+'.npy'),mode='w+',dtype=np.float64,shape=shape)
    # All history is replayed for newly added stocks and the optional fields.
    # Existing production financial columns are replaced only in the tail.
    for i,(_day,values) in enumerate(iter_financial_fields(dates,len(codes),core)):
        for name in FINANCIAL_PANEL_FIELDS:
            if i>=n0:arrays[name][i]=values[name]
            else:arrays[name][i,s0:]=values[name][s0:]
        for name in EXTRA_FIELDS:
            if i>=n0 or name not in inherited_fields:arrays[name][i]=values[name]
            else:arrays[name][i,s0:]=values[name][s0:]
        if (i+1)%1000==0:print({'financial_replay_rows':i+1,'total':len(dates)},flush=True)
    pershare=normalize_financial_events(frames['PershareIndex'],codes,tuple(PERSHARE_FIELDS.values()))
    for name in PERSHARE_FIELDS:
        if 'pit_'+name not in arrays:arrays['pit_'+name]=np.lib.format.open_memmap(fields_dir/('pit_'+name+'.npy'),mode='w+',dtype=np.float64,shape=shape)
    for i,values in enumerate(replay_latest_fields(dates,len(codes),pershare)):
        for col,name in enumerate(PERSHARE_FIELDS):
            if i>=n0 or 'pit_'+name not in inherited_fields:arrays['pit_'+name][i]=values[:,col]
            else:arrays['pit_'+name][i,s0:]=values[s0:,col]
    # Capital has share units; retain historical data, and only apply observed
    # new dated values after their source announcement, in the native WAN unit.
    from offline_data.capital_history import apply_capital_events
    capital_audit=apply_capital_events(arrays['total_share'],dates,codes,n0,s0,frames['Capital'],arrays['issue_date'])
    hashes={}
    for name,array in arrays.items():array.flush();array._mmap.close();hashes[name]=file_sha256(fields_dir/(name+'.npy'))
    np.save(fields_dir/'trade_dates.npy',dates,allow_pickle=False);np.save(fields_dir/'stock_codes.npy',np.asarray(codes),allow_pickle=False)
    target=output/f"runtime_{str(dates[0])[:4]}_{str(dates[-1]).replace('-','')}_research.npz";temporary=target.with_suffix('.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
        for path in sorted(fields_dir.glob('*.npy')):archive.write(path,path.name)
    replace_file(temporary,target)
    manifest={'state':'built_pending_independent_audit','base_path':str(Path(base_path).resolve()),'base_sha256':file_sha256(base_path),
              'runtime_path':str(target.resolve()),'runtime_sha256':file_sha256(target),'dates':[str(dates[0]),str(dates[-1])],
              'shape':list(shape),'added_stocks':list(codes[s0:]),'field_sha256':hashes,'financial_inputs':input_hashes,
              'financial_audits':{t:e.audit for t,e in core.items()},'pershare_audit':pershare.audit,'pershare_field_notes':FIELD_NOTES,'financial_chain':financial_chain,
              'capital_audit':capital_audit,'elapsed_seconds':time.monotonic()-begun,
              'limitations':['Native zero-recovery delisting convention preserved; merger proceeds are not reconstructed.',
                             'Vendor announcement vintages are not a whole-database original-filing certification.',
                             'Tail volume is source shares; legacy mixed units are resolved in the factor layer.',
                             f'Final evaluation mark is the {dates[-1]} open, not a later close.']}
    atomic_write_json(output/'manifest.json',manifest,allow_nan=False)
    atomic_write_json(output/'status.json',{'state':'built_pending_audit','elapsed_seconds':time.monotonic()-begun})
    print({'state':'built_pending_audit','runtime':str(target),'shape':shape},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--base',type=Path,required=True)
    args=parser.parse_args();build(args.root,args.base)
