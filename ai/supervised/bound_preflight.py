"""Real full-prefix, small-stock causal checks of bound local factor factories."""
import argparse
from dataclasses import replace
import gc
import os
from pathlib import Path
import time
import traceback

if __name__ == '__main__':
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
        os.environ[key]='1'
import numpy as np
from offline_data import load_runtime_slice
from factor.research import bound_groups
from .local_preflight import altered,panel_for
from utils.atomic_file import atomic_write_json,file_sha256


def run(root,runtime_path):
    from scripts.run_research_guarded import require_guarded_job
    require_guarded_job()
    begun=time.monotonic()
    rt=load_runtime_slice(runtime_path,'2005-01-01','2025-12-31',preload_rows=100001,cache_dir=root/'runtime_cache')
    indices=np.linspace(0,rt.n_stocks-1,64,dtype=int)
    arrays={}
    for key,value in rt.data.items():
        array=np.array(value[:,indices] if value.ndim==2 else value[indices],copy=True);array.flags.writeable=False;arrays[key]=array
    small=replace(rt,stock_codes=tuple(rt.stock_codes[i] for i in indices),data=arrays)
    cuts=[int(np.searchsorted(rt.trade_dates,np.datetime64('2017-06-01'))),rt.n_dates-25]
    changes=[altered(small,{},cut)[0] for cut in cuts]
    generators=[bound_groups(source) for source in [small,*changes]]
    results=[]
    for groups in zip(*generators,strict=True):
        if not all([d.metadata.name for d in group]==[d.metadata.name for d in groups[0]] for group in groups):
            raise ValueError('bound factory inventories differ across perturbations')
        for definitions in zip(*groups,strict=True):
            row={'name':definitions[0].metadata.name,'version':definitions[0].metadata.version}
            try:
                values=[np.asarray(definition.implementation().calc_batch(panel_for(source,definition,{})),dtype=float)
                        for definition,source in zip(definitions,[small,*changes],strict=True)]
                if any(value.shape!=(small.n_dates,small.n_stocks) for value in values):raise ValueError('bound output shape')
                finite=int(np.isfinite(values[0][:cuts[-1]+1]).sum())
                equal=all(np.allclose(values[0][:cut+1],other[:cut+1],rtol=1e-10,atol=1e-10,equal_nan=True)
                          for cut,other in zip(cuts,values[1:],strict=True))
                row.update(state='pass' if equal and finite else 'needs_review',finite_fixture_cells=finite)
            except Exception as exc:row.update(state='error',error=repr(exc),traceback=traceback.format_exc())
            results.append(row)
        atomic_write_json(root/'bound_preflight_partial.json',results,allow_nan=False)
        print({'completed':len(results),'elapsed':time.monotonic()-begun},flush=True);gc.collect()
    report={'state':'completed_audit','results':results,'stock_indices':indices.tolist(),'dates':rt.n_dates,
            'cutoff_dates':[str(rt.trade_dates[i]) for i in cuts],'runtime_sha256':file_sha256(runtime_path),
            'source_sha256':{p:file_sha256(p) for p in ['ai/supervised/bound_preflight.py','ai/supervised/local_preflight.py',
                'factor/research.py','offline_data/volume_units.py']},'elapsed_seconds':time.monotonic()-begun}
    atomic_write_json(root/'bound_preflight.json',report,allow_nan=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True)
    a=p.parse_args();run(a.root,a.runtime)
