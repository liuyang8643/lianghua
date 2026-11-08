"""Small-stock, full-calendar causal audit of the opt-in local catalogue."""
from __future__ import annotations
import argparse
from dataclasses import replace
import gc
import os
from pathlib import Path
import time
import traceback
import warnings

if __name__ == '__main__':
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
        os.environ[key]='1'
import numpy as np

from offline_data import load_runtime_slice
from factor.research import plain_catalog,financial_adaptations
from factor.library.local_variants import definitions as variant_definitions
from factor.library.qlib_open import lag
from utils.atomic_file import atomic_write_json,file_sha256


def panel_for(runtime,definition,pit):
    panel={**runtime.data,**{'pit_'+k:v for k,v in pit.items()},
           'trade_dates':runtime.trade_dates,'stock_codes':np.asarray(runtime.stock_codes)}
    if not definition.raw_runtime_view:
        for name in definition.metadata.required_fields:
            if name not in ('st_mask','delisted_mask','listing_age','trade_dates','stock_codes'):
                panel[name]=np.asarray(panel[name],dtype=np.float64)
    for name in definition.metadata.lagged_fields:panel[name]=lag(panel[name],1)
    return panel


def altered(runtime,pit,cutoff):
    data={}
    for key,array in runtime.data.items():
        out=np.array(array,copy=True)
        if out.ndim==2 and out.dtype.kind=='f':
            out[cutoff+1:]=out[cutoff+1:]*1.193+.317
            if key in ('high','low','close','volume','amount','total_share'):
                out[cutoff]=out[cutoff]*1.371+.531
        if key=='issue_price':
            future=runtime.field('issue_date')>runtime.trade_dates[cutoff]
            out[future]=out[future]*1.31+1
        out.flags.writeable=False;data[key]=out
    changed_pit={k:np.array(v,copy=True) for k,v in pit.items()}
    for out in changed_pit.values():out[cutoff+1:]=out[cutoff+1:]*1.19+.31;out.flags.writeable=False
    return replace(runtime,data=data),changed_pit


def run(root,runtime_path):
    from scripts.run_research_guarded import require_guarded_job
    require_guarded_job()
    begun=time.monotonic()
    runtime=load_runtime_slice(runtime_path,'2005-01-01','2025-12-31',preload_rows=100001,cache_dir=root/'runtime_cache')
    columns=np.linspace(0,runtime.n_stocks-1,64,dtype=int)
    data={}
    for key,arr in runtime.data.items():
        v=np.array(arr[:,columns] if arr.ndim==2 else arr[columns],copy=True);v.flags.writeable=False;data[key]=v
    small=replace(runtime,stock_codes=tuple(runtime.stock_codes[i] for i in columns),data=data)
    pit={}
    for path in sorted((root/'pit_pershare_ic').glob('*.npy')):
        values=np.load(path,mmap_mode='r');view=np.array(values[:,columns],copy=True);view.flags.writeable=False;pit[path.stem]=view
    if len(pit)!=7:raise ValueError('strict PIT fields incomplete')
    cutoffs=[int(np.searchsorted(runtime.trade_dates,np.datetime64('2017-06-01'))),len(runtime.trade_dates)-25]
    perturbations=[altered(small,pit,cutoff) for cutoff in cutoffs]
    definitions,excluded=plain_catalog(Path('.'),set(runtime.data))
    definitions=[*definitions,*financial_adaptations(),*variant_definitions()]
    names=[d.metadata.name for d in definitions]
    if len(names)!=len(set(names)):raise ValueError('duplicate local catalogue names')
    results=[]
    for definition in definitions:
        name=definition.metadata.name
        row={'name':name,'metadata':definition.metadata.as_dict()}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                original=np.asarray(definition.implementation().calc_batch(panel_for(small,definition,pit)),dtype=np.float64)
                changed=[np.asarray(definition.implementation().calc_batch(panel_for(rt,definition,fields)),dtype=np.float64) for rt,fields in perturbations]
            if original.shape!=(small.n_dates,small.n_stocks):raise ValueError('factor output shape mismatch')
            finite=int(np.isfinite(original[:cutoffs[-1]+1]).sum())
            passing=[]
            for delay in range(6):
                reference=lag(original,delay)
                if all(np.allclose(reference[:cut+1],lag(value,delay)[:cut+1],rtol=1e-10,atol=1e-10,equal_nan=True)
                       for cut,value in zip(cutoffs,changed,strict=True)):
                    passing.append(delay);break
            row.update(state='pass' if passing and finite else 'needs_structural_review',
                       required_output_lag=passing[0] if passing else None,finite_fixture_cells=finite)
        except Exception as exc:
            row.update(state='calculation_error',error=repr(exc),traceback=traceback.format_exc())
        results.append(row)
        if len(results)%20==0 or len(results)==len(definitions):
            status={'completed':len(results),'total':len(definitions),'elapsed_seconds':time.monotonic()-begun}
            atomic_write_json(root/'local_preflight_status.json',status,allow_nan=False)
            atomic_write_json(root/'local_preflight_partial.json',results,allow_nan=False)
            print(status,flush=True)
        gc.collect()
    report={'state':'completed_audit','scope':'unbound local definitions and existing 175 variants; bound factories audited separately',
            'source':file_sha256(Path(__file__)),'stocks':small.n_stocks,'dates':small.n_dates,
            'cutoff_dates':[str(small.trade_dates[x]) for x in cutoffs],
            'results':results,'catalog_exclusions':excluded,'elapsed_seconds':time.monotonic()-begun}
    atomic_write_json(root/'local_preflight.json',report,allow_nan=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True)
    args=p.parse_args();run(args.root,args.runtime)
