"""Full local factor IC, with explicit reuse of compatible historical results."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import time
import traceback
import warnings

if __name__ == '__main__':
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
        os.environ[key]='1'
import numpy as np
import psutil
from offline_data import load_runtime_slice
from factor.research import plain_catalog,bound_groups,financial_adaptations
from factor.library.local_variants import Inputs,candidates,definitions as variant_definitions,variant_name
from env.forward_returns import open_forward_returns,label_split_mask
from .local_preflight import panel_for
from .factor_stage import validate_record
from .ic import daily_ic,summary,pair_counts_and_constants
from utils.atomic_file import atomic_write_json,file_sha256


def run(root,runtime_path,*,stage_name='local_factors_v2'):
    from scripts.run_research_guarded import require_guarded_job
    require_guarded_job()
    started=time.monotonic();stage=root/stage_name;stage.mkdir(exist_ok=True)
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    legacy=Path(protocol['legacy_ic_root'])
    paths=[Path(__file__),Path('ai/supervised/local_preflight.py'),Path('ai/supervised/ic.py'),
           Path('ai/supervised/factor_stage.py'),Path('offline_data/runtime.py'),Path('offline_data/volume_units.py'),
           root/'protocol.json',root/'local_preflight.json',root/'bound_preflight.json',root/'pit_pershare_ic/identity.json',
           legacy/'protocol.json',legacy/'dates.npy',legacy/'eligible.npy',*(legacy/f'final_labels_{h}d.npy' for h in (1,5,20))]
    paths.extend(sorted(Path('factor').rglob('*.py')));paths.extend(sorted(Path('factor_db/factors').glob('*.py')))
    identity={'sources':{str(p):file_sha256(p) for p in paths},'runtime_sha256':file_sha256(runtime_path)}
    identity_path=stage/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text(encoding='utf-8'))!=identity:
        raise ValueError('local IC source/input identity changed; do not reuse old numeric records')
    atomic_write_json(identity_path,identity,allow_nan=False)
    rt=load_runtime_slice(runtime_path,*protocol['ic_comparison_period'],preload_rows=100001,cache_dir=root/'runtime_cache')
    a,b=rt.decision_start,rt.decision_stop;dates=rt.trade_dates[a:b]
    eligible=(rt.field('listing_age')[a:b]>=0)&~rt.field('delisted_mask')[a:b]&~rt.field('st_mask')[a:b]&(rt.field('open')[a:b]>=2)
    np.testing.assert_array_equal(dates,np.load(legacy/'dates.npy'));np.testing.assert_array_equal(eligible,np.load(legacy/'eligible.npy',mmap_mode='r'))
    pit_receipt=json.loads((root/'pit_pershare_ic/identity.json').read_text(encoding='utf-8'))
    pit={}
    for name,digest in pit_receipt['array_sha256'].items():
        path=root/'pit_pershare_ic'/(name+'.npy')
        if file_sha256(path)!=digest:raise ValueError('PIT field bytes changed')
        pit[name]=np.load(path,mmap_mode='r')
        if pit[name].shape!=(rt.n_dates,rt.n_stocks):raise ValueError('PIT axes differ')
    labels={h:np.load(legacy/f'final_labels_{h}d.npy',mmap_mode='r') for h in (1,5,20)}
    label10=stage/'labels_10d.npy'
    expected=open_forward_returns(rt.data,10)[a:b]
    if label10.exists():np.testing.assert_allclose(np.load(label10,mmap_mode='r'),expected,rtol=1e-12,atol=1e-12,equal_nan=True)
    else:np.save(label10,expected,allow_pickle=False)
    del expected;labels[10]=np.load(label10,mmap_mode='r')
    for h in (1,5,20):
        expected=open_forward_returns(rt.data,h)[a:b]
        np.testing.assert_allclose(expected[eligible],labels[h][eligible],rtol=1e-12,atol=1e-12,equal_nan=True);del expected
    np.save(stage/'dates.npy',dates,allow_pickle=False);np.save(stage/'eligible.npy',eligible,allow_pickle=False)
    for folder in ('scores','daily','records'):(stage/folder).mkdir(exist_ok=True)
    audits={x['name']:x for file in ('local_preflight.json','bound_preflight.json')
            for x in json.loads((root/file).read_text(encoding='utf-8'))['results']}
    records=[];seen=set();by_score={}

    def process(definition,provided=None):
        name=definition.metadata.name
        if name in seen:return
        seen.add(name)
        if name not in audits or audits[name]['state']!='pass':raise ValueError('factor lacks passing causal fixture: '+name)
        dest=stage/'records'/f'{name}.json'
        if dest.exists():
            record=json.loads(dest.read_text(encoding='utf-8'))
            if record['metadata']!=definition.metadata.as_dict():raise ValueError('factor implementation metadata changed')
            validate_record(record,eligible.shape,dates,protocol);records.append(record);by_score[record['scores_sha256']]=record;return
        if (stage/'STOP').exists() or psutil.virtual_memory().available<8*2**30:raise RuntimeError('STOP or low available memory')
        began=time.monotonic()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            raw=definition.implementation().calc_batch(panel_for(rt,definition,pit)) if provided is None else provided
        raw=np.asarray(raw,dtype=np.float64)
        if raw.shape!=(rt.n_dates,rt.n_stocks):raise ValueError('factor shape differs: '+name)
        values=np.ascontiguousarray(raw[a:b]);del raw
        if np.isinf(values).any():values=np.where(np.isfinite(values),values,np.nan)
        # Compatibility is exact on every eligible cell, including NaNs.
        # Differences outside the trading universe never justify changing IC.
        old_name=name.removeprefix('Optimized__');old_scores=legacy/'scores'/f'{old_name}.npy'
        compatible=False
        if old_scores.exists():
            old=np.load(old_scores,mmap_mode='r')
            compatible=old.shape==values.shape and np.array_equal(values[eligible],old[eligible],equal_nan=True)
            del old
        if compatible:score_path=old_scores
        else:
            score_path=stage/'scores'/f'{name}.npy';tmp=score_path.with_suffix('.tmp')
            with tmp.open('wb') as stream:np.save(stream,values,allow_pickle=False)
            os.replace(tmp,score_path)
        score_hash=file_sha256(score_path);reuse=[]
        if score_hash in by_score:
            prior=by_score[score_hash]
            daily_path=Path(prior['daily_path']);daily_hash=prior['daily_sha256'];statistics=prior['statistics']
            numeric_alias=prior['name'];reuse=['current_identical_numeric_factor']
        else:
            arrays={};statistics={};numeric_alias=None
            for h in (1,5,10,20):
                old_path=legacy/'final_daily'/f'{old_name}.npz'
                if compatible and h in (1,5,20) and old_path.exists():
                    with np.load(old_path,allow_pickle=False) as z:daily=z[f'raw_{h}d'].copy()
                    counts,constant=pair_counts_and_constants(values,labels[h],eligible)
                    np.testing.assert_array_equal(daily[:,2],counts)
                    daily[(counts<30)|constant,:2]=np.nan;reuse.append(h)
                else:daily=daily_ic(values,labels[h],eligible)
                arrays[str(h)]=daily;statistics[str(h)]={'train':summary(daily,label_split_mask(dates,*protocol['train'],h))}
            daily_path=stage/'daily'/f'{name}.npz';tmp=daily_path.with_suffix('.tmp')
            with tmp.open('wb') as stream:np.savez_compressed(stream,**arrays)
            os.replace(tmp,daily_path);daily_hash=file_sha256(daily_path)
        record={'name':name,'metadata':definition.metadata.as_dict(),'status':'computed',
                'scores_path':str(score_path),'scores_sha256':score_hash,'daily_path':str(daily_path),'daily_sha256':daily_hash,
                'statistics':statistics,'reused_ic_horizons':reuse,'legacy_raw_eligible_exact':compatible,
                'numeric_alias_of':numeric_alias,'causal_fixture':audits[name],
                'coverage':float((np.isfinite(values)&eligible).sum()/eligible.sum()),'elapsed_seconds':time.monotonic()-began}
        atomic_write_json(dest,record,allow_nan=False);records.append(record);by_score[score_hash]=record
        status={'state':'computing','completed':len(records),'factor':name,'elapsed_seconds':time.monotonic()-started,
                'available_gib':psutil.virtual_memory().available/2**30,'reused_ic_horizons':reuse}
        atomic_write_json(stage/'status.json',status,allow_nan=False);print(status,flush=True)
        del values;gc.collect()

    definitions,excluded=plain_catalog(Path('.'),set(rt.data))
    for definition in [*definitions,*financial_adaptations()]:
        if definition.metadata.name not in ('BiliAdjustedIssueDiscount','BiliHighLifetimeRangeRatio'):process(definition)
    for group in bound_groups(rt):
        for definition in group:process(definition)
    inputs=Inputs({**rt.data,'trade_dates':rt.trade_dates,'stock_codes':rt.stock_codes})
    variants={x.metadata.name:x for x in variant_definitions()}
    for name,parameters in candidates():
        key=variant_name(name,parameters)
        if (stage/'records'/f'{key}.json').exists():process(variants[key]);continue
        process(variants[key],inputs.score(name,parameters))
    atomic_write_json(stage/'inventory.json',records,allow_nan=False);atomic_write_json(stage/'exclusions.json',excluded,allow_nan=False)
    atomic_write_json(stage/'status.json',{'state':'complete','scope':'local daily IC comparison only','completed':len(records),'elapsed_seconds':time.monotonic()-started},allow_nan=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True)
    a=p.parse_args();run(a.root,a.runtime)
