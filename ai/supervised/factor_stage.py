"""Checkpointed, offline Qlib feature and IC inventory. No model or trading."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
import shutil
from pathlib import Path
import time

if __name__ == '__main__':
    for _thread_var in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[_thread_var]='1'

import numpy as np
import psutil

from offline_data import load_runtime_slice
from utils.atomic_file import file_sha256
from factor.library.qlib_open import CATALOG, ExpressionPanel, price_fields
from env.forward_returns import open_forward_returns, label_split_mask
from .ic import daily_ic,summary


def save_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)


def validate_record(record, expected_shape, dates, protocol):
    """Never treat a JSON row alone as proof of completed numeric work."""
    for key in ('scores','daily'):
        path=Path(record[key+'_path'])
        if not path.is_file() or file_sha256(path)!=record[key+'_sha256']:
            raise ValueError(f'{key} output missing or content hash mismatch')
    values=np.load(record['scores_path'],mmap_mode='r',allow_pickle=False)
    if values.shape!=expected_shape or values.dtype!=np.float64:
        raise ValueError('factor score artifact shape or dtype mismatch')
    with np.load(record['daily_path'],allow_pickle=False) as payload:
        if set(payload.files)!={'1','5','10','20'}:
            raise ValueError('daily IC horizons mismatch')
        for h in (1,5,10,20):
            daily=payload[str(h)]
            if daily.shape!=(len(dates),3):raise ValueError('daily IC artifact shape mismatch')
            expected={'train':summary(daily,label_split_mask(dates,*protocol['train'],h))}
            if expected!=record['statistics'][str(h)]:
                raise ValueError('saved IC statistics differ from the declared training mask')


def run(root, runtime_path, *, stage_name='qlib_factors_v2'):
    from scripts.run_research_guarded import require_guarded_job
    require_guarded_job()
    start=time.monotonic()
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    stage=root/stage_name;stage.mkdir(exist_ok=True)
    legacy=Path(protocol['legacy_ic_root'])
    source_files=[Path(__file__),Path('factor/library/qlib_open.py'),Path('factor/library/qlib_catalog.json'),Path('offline_data/volume_units.py'),Path('env/forward_returns.py'),Path('ai/supervised/ic.py'),Path('offline_data/runtime.py'),
                  root/'protocol.json',legacy/'dates.npy',legacy/'eligible.npy',*(legacy/f'final_labels_{h}d.npy' for h in (1,5,20))]
    identity={str(p):file_sha256(p) for p in source_files}
    source_id=file_sha256(runtime_path)
    if (stage/'identity.json').exists():
        old=json.loads((stage/'identity.json').read_text())
        if old != {'source':identity,'runtime_sha256':source_id}:
            raise RuntimeError('stage source changed; existing numeric results must not be silently reused')
    else: save_json(stage/'identity.json',{'source':identity,'runtime_sha256':source_id})
    identity_sha=hashlib.sha256(json.dumps({'source':identity,'runtime_sha256':source_id},sort_keys=True).encode()).hexdigest()
    for path in source_files:
        if path.suffix=='.py':
            target=stage/'source'/path.resolve().relative_to(Path.cwd().resolve())
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copy2(path,target)
            if file_sha256(target)!=identity[str(path)]:raise ValueError('source capsule mismatch')
    save_json(stage/'status.json',{'state':'loading','pid':os.getpid()})
    comparison_start,comparison_end=protocol['ic_comparison_period']
    rt=load_runtime_slice(runtime_path,comparison_start,comparison_end,preload_rows=10000,cache_dir=root/'runtime_cache')
    first,stop=rt.decision_start,rt.decision_stop
    dates=rt.trade_dates[first:stop]
    eligible=(rt.field('listing_age')[first:stop]>=0)&~rt.field('delisted_mask')[first:stop]&~rt.field('st_mask')[first:stop]&(rt.field('open')[first:stop]>=2)
    np.save(stage/'dates.npy',dates,allow_pickle=False);np.save(stage/'eligible.npy',eligible,allow_pickle=False)
    np.testing.assert_array_equal(dates,np.load(legacy/'dates.npy',allow_pickle=False))
    np.testing.assert_array_equal(eligible,np.load(legacy/'eligible.npy',mmap_mode='r'))
    labels={h:np.load(legacy/f'final_labels_{h}d.npy',mmap_mode='r') for h in (1,5,20)}
    path=stage/'labels_10d.npy'
    if not path.exists(): np.save(path,open_forward_returns(rt.data,10)[first:stop],allow_pickle=False)
    labels[10]=np.load(path,mmap_mode='r')
    actual=open_forward_returns(rt.data,10)[first:stop]
    np.testing.assert_allclose(actual,labels[10],atol=1e-12,rtol=1e-12,equal_nan=True)
    del actual
    save_json(stage/'scope.json',{'scope':'factor IC comparison and train-only screening; not the final model test-feature cache',
                                'period':protocol['ic_comparison_period'],'model_test_period_separate':protocol['test'],
                                'label_10d_sha256':file_sha256(path),'identity_sha256':identity_sha})
    # Existing label convention parity, including terminal-over-missing rule.
    for h in (1,5,20):
        actual=open_forward_returns(rt.data,h)[first:stop]
        # Legacy labels retain absorbing-zero returns after delisting, whereas
        # the public label function masks nonmembers. Neither can enter IC.
        # Compare every eligible cell, including missing outcomes, not only
        # their finite intersection (which could conceal missing-label drift).
        np.testing.assert_allclose(actual[eligible],labels[h][eligible],atol=1e-12,rtol=1e-12,equal_nan=True)
        del actual
    fields=price_fields(rt.data)
    evaluator=ExpressionPanel(fields,cache_entries=3)
    (stage/'scores').mkdir(exist_ok=True);(stage/'daily').mkdir(exist_ok=True);(stage/'records').mkdir(exist_ok=True)
    aliases={};records=[]
    for item in CATALOG['features']:
        name=item['name']; record_path=stage/'records'/f'{name}.json'
        if record_path.exists():
            existing=json.loads(record_path.read_text(encoding='utf-8'))
            if existing.get('input_identity_sha256')!=identity_sha or existing['expression']!=item['expression']:
                raise ValueError('factor record identity mismatch')
            validate_record(existing,eligible.shape,dates,protocol)
            records.append(existing)
            aliases.setdefault(item['expression'],name)
            continue
        if (root/'STOP').exists() or psutil.virtual_memory().available<6*2**30:
            raise RuntimeError('STOP or available memory below 6GiB; completed columns retained')
        begun=time.monotonic()
        if item['expression'] in aliases:
            canonical=aliases[item['expression']]
            rec={**item,'alias_of':canonical,'status':'exact_expression_alias'}
            original=json.loads((stage/'records'/f'{canonical}.json').read_text(encoding='utf-8'))
            rec.update({k:original[k] for k in ['scores_path','daily_path','scores_sha256','daily_sha256','statistics','coverage']})
        else:
            values=evaluator.evaluate(item['expression'])[first:stop]
            score_path=stage/'scores'/f'{name}.npy'
            temp=score_path.with_suffix('.tmp')
            with temp.open('wb') as stream: np.save(stream,values,allow_pickle=False)
            os.replace(temp,score_path)
            stats={};arrays={}
            for h in (1,5,10,20):
                daily=daily_ic(values,labels[h],eligible)
                arrays[str(h)]=daily
                # Summary selection endpoint is strictly training-contained.
                stats[str(h)]={'train':summary(daily,label_split_mask(dates,*protocol['train'],h))}
            daily_path=stage/'daily'/f'{name}.npz';temp=daily_path.with_suffix('.tmp')
            with temp.open('wb') as stream:np.savez_compressed(stream,**arrays)
            os.replace(temp,daily_path)
            coverage=float((np.isfinite(values)&eligible).sum()/eligible.sum())
            rec={**item,'status':'computed','scores_path':str(score_path),'daily_path':str(daily_path),
                 'scores_sha256':file_sha256(score_path),'daily_sha256':file_sha256(daily_path),
                 'statistics':stats,'coverage':coverage,'elapsed_seconds':time.monotonic()-begun}
            aliases[item['expression']]=name
            del values,arrays
        rec['input_identity_sha256']=identity_sha
        save_json(record_path,rec);records.append(rec)
        status={'state':'computing','completed':len(records),'total':len(CATALOG['features']),'factor':name,'elapsed_seconds':time.monotonic()-start,
                'pid':os.getpid(),'available_gib':psutil.virtual_memory().available/2**30}
        save_json(stage/'status.json',status);print(json.dumps(status),flush=True)
        gc.collect()
    save_json(stage/'inventory.json',records)
    save_json(stage/'status.json',{'state':'complete','scope':'IC comparison only','completed':len(records),'elapsed_seconds':time.monotonic()-start,'pid':os.getpid()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--runtime',type=Path,required=True)
    args=parser.parse_args();run(args.root,args.runtime)
