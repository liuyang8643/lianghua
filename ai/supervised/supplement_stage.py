"""Complete the data-backed local supplements without modifying running batches."""
import argparse
import json
import os
from pathlib import Path
import time

if __name__ == '__main__':
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
        os.environ[key]='1'
import numpy as np
import pandas as pd

from offline_data import load_runtime_slice
from offline_data.supplemental_fields import industry_panel,implemented_dividends
from factor.library.research_supplements import definitions,cash_dividend_scores,monthly_pulse
from env.forward_returns import open_forward_returns,label_split_mask
from .ic import daily_ic,summary
from utils.atomic_file import file_sha256,atomic_write_json,replace_file
from .monthly_data import verified_monthly_path


def run(root,runtime_path):
    from scripts.run_research_guarded import require_guarded_job
    require_guarded_job()
    start=time.monotonic();stage=root/'supplement_factors';stage.mkdir(exist_ok=True)
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'));legacy=Path(protocol['legacy_ic_root'])
    monthly_path=verified_monthly_path(protocol)
    rt=load_runtime_slice(runtime_path,*protocol['ic_comparison_period'],preload_rows=100001,cache_dir=root/'runtime_cache')
    a,b=rt.decision_start,rt.decision_stop;dates=rt.trade_dates[a:b]
    eligible=np.load(legacy/'eligible.npy',mmap_mode='r');np.testing.assert_array_equal(dates,np.load(legacy/'dates.npy'))
    labels={h:np.load(legacy/f'final_labels_{h}d.npy',mmap_mode='r') for h in (1,5,20)}
    labels[10]=open_forward_returns(rt.data,10)[a:b]
    ext=json.loads((root/'extended_runtime/manifest.json').read_text(encoding='utf-8'))
    sources=[Path(__file__),Path('factor/library/research_supplements.py'),Path('offline_data/supplemental_fields.py'),
             Path('factor_db/factors/IndustryPeerReversalStrict.py'),root/'protocol.json',root/'extended_runtime/manifest.json',
             Path('data/industry_history/sw_industry_history.parquet'),root/'dividend_source/events.parquet',
             monthly_path,monthly_path.with_name('manifest.json')]
    identity={str(p):file_sha256(p) for p in sources};receipt=stage/'identity.json'
    if receipt.exists() and json.loads(receipt.read_text())!=identity:raise ValueError('supplement identity changed')
    atomic_write_json(receipt,identity)
    for folder in ('scores','daily','records'):(stage/folder).mkdir(exist_ok=True)
    panel={**rt.data,'trade_dates':rt.trade_dates,'stock_codes':rt.stock_codes}
    extra={field for d in definitions() for field in d.metadata.required_fields}-set(panel)-{'industry_id'}
    for field in extra:
        path=root/'extended_runtime/fields'/(field+'.npy')
        if file_sha256(path)!=ext['field_sha256'][field]:raise ValueError('extra financial field changed')
        panel[field]=np.load(path,mmap_mode='r')[:rt.n_dates,:rt.n_stocks]
    history=pd.read_parquet('data/industry_history/sw_industry_history.parquet')
    panel['industry_id']=industry_panel(history,rt.trade_dates,rt.stock_codes,(rt.field('listing_age')>=0)&~rt.field('delisted_mask'))
    np.save(stage/'industry_id.npy',panel['industry_id'],allow_pickle=False)
    records=[]
    def process(name,values,metadata,cadence='daily'):
        score=np.asarray(values[a:b],dtype=np.float64);score_path=stage/'scores'/f'{name}.npy'
        temp=score_path.with_suffix('.tmp')
        with temp.open('wb') as stream:np.save(stream,score,allow_pickle=False)
        replace_file(temp,score_path)
        arrays={};stats={};reuse=[]
        for h in (1,5,10,20):
            daily=daily_ic(score,labels[h],eligible)
            arrays[str(h)]=daily;stats[str(h)]={'train':summary(daily,label_split_mask(dates,*protocol['train'],h))}
        daily_path=stage/'daily'/f'{name}.npz';np.savez_compressed(daily_path,**arrays)
        record={'name':name,'metadata':metadata,'cadence':cadence,'scores_path':str(score_path),'scores_sha256':file_sha256(score_path),
                'daily_path':str(daily_path),'daily_sha256':file_sha256(daily_path),'statistics':stats,'reused_ic_horizons':reuse,
                'coverage':float((np.isfinite(score)&eligible).sum()/eligible.sum())}
        atomic_write_json(stage/'records'/f'{name}.json',record,allow_nan=False);records.append(record)
        atomic_write_json(stage/'status.json',{'state':'computing','completed':len(records),'name':name,'elapsed_seconds':time.monotonic()-start})
        print({'completed':len(records),'name':name},flush=True)
    for definition in definitions():process(definition.metadata.name,definition.implementation().calc_batch(panel),definition.metadata.as_dict())
    events,report=implemented_dividends(pd.read_parquet(root/'dividend_source/events.parquet'))
    atomic_write_json(stage/'dividend_normalization.json',report)
    for name,value in cash_dividend_scores(panel,events).items():process(name,value,{'version':'implemented-dividends-full-shsz-v1','source_sha256':identity[str(root/'dividend_source/events.parquet')]})
    with np.load(monthly_path,allow_pickle=False) as monthly:
        for name in monthly['names'].astype(str):process(name,monthly_pulse(rt.trade_dates,rt.stock_codes,monthly,name),
            {'version':'monthly-next-open-pulse-asof-shares-v2','monthly_sha256':file_sha256(monthly_path)},cadence='monthly_no_carry')
    atomic_write_json(stage/'inventory.json',records,allow_nan=False)
    atomic_write_json(stage/'status.json',{'state':'complete','completed':len(records),'elapsed_seconds':time.monotonic()-start})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True)
    x=p.parse_args();run(x.root,x.runtime)
