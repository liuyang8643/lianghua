"""Search helpers and the train-only entry point; legacy code is audit-only."""
import argparse
from dataclasses import asdict
import gc
import itertools
import json
import os
from pathlib import Path
import random
import time

if __name__=='__main__':
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import numpy as np
from env.fees import DEFAULT_FEE_SCHEDULE
from .portfolio import PreparedModelPortfolio
from utils.atomic_file import atomic_write_json,file_sha256

TOPK_SPACE={'topk':[20,30,50,100],'n_drop':[1,2,5,10],'hold_thresh':[1,3,5,10]}
MVO_SPACE={'pool_size':[20,40,80],'risk_aversion':[.1,1.,10.,100.],'turnover_limit':[.02,.05,.1,.2,.4,.8,1.,2.],
           'ridge':[0.,.0001,.001,.01],'lookback':[60,120,252],'shrinkage':[.1,.5,.9]}


def key(parameters):return json.dumps(parameters,sort_keys=True,separators=(',',':'))


def ga_proposals(space,completed,rng,population=20):
    keys=list(space);valid=[r for r in completed if r.get('state')=='complete']
    if len(valid)<population:
        return {name:rng.choice(space[name]) for name in keys}
    elite=sorted(valid,key=lambda r:r['metrics']['calmar'],reverse=True)[:population]
    def parent():return max(rng.sample(elite,min(3,len(elite))),key=lambda r:r['metrics']['calmar'])['parameters']
    a,b=parent(),parent()
    child={name:(a[name] if rng.random()<.5 else b[name]) for name in keys}
    for name in keys:
        if rng.random()<.25:child[name]=rng.choice(space[name])
    return child


def rank_trial(row):
    m=row['metrics'];return (m['calmar'],m['annualized_return'],-m['max_drawdown'],-m['mean_gross_turnover'])


def portfolio_methods(protocol):
    methods=tuple(protocol.get('portfolio_methods',('topk','mvo')))
    if not methods or len(set(methods))!=len(methods) or any(x not in ('topk','mvo') for x in methods):
        raise ValueError('portfolio_methods must contain distinct supported strategies')
    return methods


def run(root):
    protocol=json.loads((Path(root)/'protocol.json').read_text(encoding='utf-8'))
    if protocol.get('selection_split')!='train':raise ValueError('validation-based optimization is disabled; use a new training-only study')
    from .train_only_search import run as training_search
    return training_search(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
