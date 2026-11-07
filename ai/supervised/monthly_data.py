"""Rebuild archived monthly source signals with the current as-of adapter."""
import argparse
import gzip
import json
from pathlib import Path
import time
import numpy as np
from offline_data import load_runtime_slice
from offline_data.volume_units import canonical_volume
from offline_data.sources.sina_research_history import align_float_shares,align_main_flow,parse_share_structure
from factor.library.bilibili_money_flow import NAMES,completed_close_stock_scores
from utils.atomic_file import atomic_write_json,file_sha256

MONTHLY_DEPENDENCIES=('offline_data/sources/sina_research_history.py','factor/library/bilibili_money_flow.py','offline_data/volume_units.py','ai/supervised/monthly_data.py')

def verified_monthly_path(protocol):
    path=Path(protocol['monthly_signal_path'])
    manifest=json.loads(path.with_name('manifest.json').read_text(encoding='utf-8'))
    if manifest['state']!='complete' or file_sha256(path)!=manifest['output_sha256']:
        raise ValueError('monthly source signal is incomplete or changed')
    for name in MONTHLY_DEPENDENCIES:
        if file_sha256(name)!=manifest['implementation_sha256'].get(name):
            raise ValueError('monthly source implementation changed; rebuild its signals')
    return path


def run(root):
    from scripts.run_research_guarded import require_guarded_job
    require_guarded_job();root=Path(root);started=time.monotonic()
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    active=json.loads((root/'active_data.json').read_text(encoding='utf-8'))
    source=Path(protocol['monthly_source_root']);terminal=json.loads((source/'terminal.json').read_text(encoding='utf-8'))
    if (source/'fetch.lock').exists() or not terminal['all_processed']:raise ValueError('monthly source acquisition incomplete')
    out=Path(protocol['monthly_signal_path']).parent;out.mkdir(exist_ok=False)
    rt=load_runtime_slice(active['runtime_path'],'2013-11-01',protocol.get('data_end') or protocol['test'][1],cache_dir=root/'monthly_runtime_cache')
    if rt.manifest.source_sha256!=active['runtime_sha256']:raise ValueError('monthly runtime changed')
    dates=rt.trade_dates;codes=rt.stock_codes;months=dates.astype('datetime64[M]')
    rows=np.flatnonzero(months[:-1]!=months[1:]);rows=rows[dates[rows]>=np.datetime64('2013-12-01')]
    scores=np.full((len(NAMES),len(rows),len(codes)),np.nan)
    eligible=(rt.field('listing_age')[rows]>=0)&~rt.field('delisted_mask')[rows]&~rt.field('st_mask')[rows]
    eligible&=(dates[rows,None]-rt.field('issue_date')[None,:])>=np.timedelta64(365,'D')
    eligible&=np.isfinite(rt.field('volume')[rows])&(rt.field('volume')[rows]>0)
    sources={};missing=[];quality=[]
    for j,code in enumerate(codes):
        folder=source/code;receipt=folder/'receipt.json'
        if not receipt.exists():missing.append(code);continue
        saved=json.loads(receipt.read_text(encoding='utf-8'))
        for name,sha in saved['files_sha256'].items():
            if file_sha256(folder/name)!=sha:raise ValueError('monthly archived source changed: '+str(folder/name))
        sources[code]={'receipt_sha256':file_sha256(receipt),'files_sha256':saved['files_sha256']}
        events=parse_share_structure(gzip.decompress((folder/'shares.raw.gz').read_bytes()),code)
        shares,known=align_float_shares(events,dates)
        with np.load(folder/'flow.npz',allow_pickle=False) as z:flow={'dates':z['dates'],'fields':z['fields'].astype(str).tolist(),'values':z['values']}
        raw_volume=rt.field('volume')[:,j];amount=rt.field('amount')[:,j]
        net,gross,audit=align_main_flow(flow,dates,raw_volume,amount)
        volume=canonical_volume({k:rt.field(k)[:,j] for k in ('volume','amount','low','high')},shares=False)
        inconsistent=((raw_volume==0)&np.isfinite(amount)&(amount>0))|np.isin(dates,np.asarray(audit['cross_provider_zero_conflict_dates'],dtype='datetime64[D]'))
        volume[inconsistent]=np.nan
        scores[:,:,j]=completed_close_stock_scores(volume,rt.field('close')[:,j],shares,known,net,gross)[:,rows]
        quality.append({'code':code,'known_share_days':int(known.sum()),**audit})
        if (j+1)%250==0:
            atomic_write_json(out/'status.json',{'state':'computing','completed':j+1,'total':len(codes),'seconds':time.monotonic()-started})
            print({'monthly_completed':j+1,'total':len(codes)},flush=True)
    np.savez_compressed(out/'monthly.npz',names=np.asarray(NAMES),stock_codes=np.asarray(codes),signal_dates=dates[rows],scores=scores,eligible=eligible)
    manifest={'state':'complete','runtime_sha256':active['runtime_sha256'],'source_terminal_sha256':file_sha256(source/'terminal.json'),
        'implementation_sha256':{p:file_sha256(p) for p in MONTHLY_DEPENDENCIES},
        'volume_rule':'completed-bar canonical lots; unknown or ambiguous units remain missing; raw volume retained only for source flow zero-consistency checks',
        'source_ids':sources,'missing_codes':missing,'quality':quality,'output_sha256':file_sha256(out/'monthly.npz'),
        'float_share_rule':'latest effective event among announcements known strictly before each closed day; future events cannot truncate earlier state',
        'signal_dates':[str(dates[rows[0]]),str(dates[rows[-1]])],'seconds':time.monotonic()-started}
    atomic_write_json(out/'manifest.json',manifest);atomic_write_json(out/'status.json',{'state':'complete','seconds':time.monotonic()-started})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);run(parser.parse_args().root)
