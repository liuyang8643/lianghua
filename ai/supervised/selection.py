"""Exact common-sample daily Spearman and per-horizon greedy factor selection."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import hashlib

if __name__ == '__main__':
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[k]='1'
import numpy as np
from numba import njit,prange,set_num_threads
from scipy.stats import rankdata
from utils.atomic_file import atomic_write_json,file_sha256
from factor_db import require_allowed_factor_names


@njit(cache=True,parallel=True,fastmath=False)
def _exact_pairs(values,valid,orders,nvalid,groups,correlations,counts):
    factors,stocks=values.shape
    for worker_index in prange(factors):
        # Alternate long and short triangular rows to balance static chunks.
        i=worker_index//2 if worker_index%2==0 else factors-1-worker_index//2
        rank_i=np.empty(stocks,np.float64)
        for j in range(i+1,factors):
            if groups[i]==groups[j]:continue
            if counts[i,j]<30:
                correlations[i,j]=np.nan;correlations[j,i]=np.nan;continue
            position=0;k=0
            while k<nvalid[i]:
                end=k+1
                while end<nvalid[i] and values[i,orders[i,end]]==values[i,orders[i,k]]:end+=1
                common=0
                for z in range(k,end):common+=valid[j,orders[i,z]]
                r=position+(common-1)*.5
                for z in range(k,end):
                    stock=orders[i,z]
                    if valid[j,stock]:rank_i[stock]=r
                position+=common;k=end
            mean=(position-1)*.5;position=0;k=0;xx=0.;yy=0.;xy=0.
            while k<nvalid[j]:
                end=k+1
                while end<nvalid[j] and values[j,orders[j,end]]==values[j,orders[j,k]]:end+=1
                common=0
                for z in range(k,end):common+=valid[i,orders[j,z]]
                r=position+(common-1)*.5-mean
                for z in range(k,end):
                    stock=orders[j,z]
                    if valid[i,stock]:
                        other=rank_i[stock]-mean;xx+=other*other;yy+=r*r;xy+=other*r
                position+=common;k=end
            value=xy/np.sqrt(xx*yy) if xx>0 and yy>0 else np.nan
            correlations[i,j]=value;correlations[j,i]=value
    return correlations


def exact_day_spearman(values):
    values=np.ascontiguousarray(values,dtype=np.float64);valid=np.isfinite(values)
    # A shared validity mask allows one BLAS product. Other pairs are exactly
    # reranked on their intersection in linear time using pre-sorted orders.
    keys={};groups=[]
    for mask in valid:
        key=mask.tobytes()
        if key not in keys:keys[key]=len(keys)
        groups.append(keys[key])
    ranks=rankdata(np.where(valid,values,np.nan),axis=1,method='average',nan_policy='omit')
    nvalid=valid.sum(axis=1).astype(np.int64)
    center=np.where(valid,ranks-(nvalid[:,None]+1)*.5,0.)
    norm=np.sqrt(np.sum(center*center,axis=1))
    normalized=np.divide(center,norm[:,None],out=np.zeros_like(center),where=norm[:,None]>0)
    correlations=normalized@normalized.T
    counts=(valid.astype(np.float64)@valid.T).astype(np.int32)
    correlations[(counts<30)|(norm[:,None]==0)|(norm[None,:]==0)]=np.nan
    orders=np.argsort(np.where(valid,values,np.inf),axis=1,kind='stable').astype(np.int32)
    result=_exact_pairs(values,valid,orders,nvalid,np.asarray(groups),correlations,counts)
    return np.clip(result,-1,1),counts


def greedy_select(names,means,correlation,threshold=.8):
    require_allowed_factor_names(names)
    order=sorted((i for i,n in enumerate(names) if means.get(n) is not None and means[n]>0),key=lambda i:(-means[names[i]],names[i]))
    keep=[];removed=[]
    for i in order:
        conflict=next((j for j in keep if np.isfinite(correlation[i,j]) and abs(correlation[i,j])>=threshold),None)
        if conflict is None:keep.append(i)
        else:removed.append({'name':names[i],'retained':names[conflict],'correlation':float(correlation[i,conflict]),'ic':means[names[i]],'retained_ic':means[names[conflict]]})
    return keep,removed


def run(root,threads=8):
    from scripts.run_research_guarded import require_guarded_job
    require_guarded_job();set_num_threads(threads);started=time.monotonic()
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'));records=[]
    sources=['qlib_factors_v2','local_factors_v2','supplement_factors']
    for source in sources:
        status=json.loads((root/source/'status.json').read_text())
        if status['state']!='complete':raise ValueError(source+' is not complete')
        records.extend(json.loads((root/source/'inventory.json').read_text(encoding='utf-8')))
    require_allowed_factor_names(r['name'] for r in records)
    by_name={}
    for record in records:
        if record['name'] in by_name:raise ValueError('duplicate factor inventory name '+record['name'])
        by_name[record['name']]=record
    means={h:{r['name']:r['statistics'][str(h)]['train']['rank_ic'] for r in records} for h in (1,5,10)}
    positive=[r for r in records if any(means[h][r['name']] is not None and means[h][r['name']]>0 for h in means)]
    # Exact numeric aliases have identical IC and correlation. Keep a name map
    # instead of repeating their pair computations; selection still sees names.
    representatives={};aliases={}
    for r in positive:
        key=r['scores_sha256']
        if key in representatives:aliases[r['name']]=representatives[key]['name']
        else:representatives[key]=r
    selected=list(representatives.values());names=[r['name'] for r in selected]
    for record in selected:
        if file_sha256(record['scores_path'])!=record['scores_sha256']:raise ValueError('factor score bytes changed: '+record['name'])
    matrices=[np.load(r['scores_path'],mmap_mode='r',allow_pickle=False) for r in selected]
    dates=np.load(Path(protocol['legacy_ic_root'])/'dates.npy');eligible=np.load(Path(protocol['legacy_ic_root'])/'eligible.npy',mmap_mode='r')
    if any(x.shape!=eligible.shape for x in matrices):raise ValueError('factor correlation axes differ')
    which=np.flatnonzero((dates>=np.datetime64(protocol['train'][0]))&(dates<=np.datetime64(protocol['train'][1])))
    stage=root/'selection';stage.mkdir(exist_ok=True)
    identity={'inventories':{s:file_sha256(root/s/'inventory.json') for s in sources},'protocol':file_sha256(root/'protocol.json'),
              'axes':{name:file_sha256(Path(protocol['legacy_ic_root'])/name) for name in ['dates.npy','eligible.npy']},
              'which_sha256':hashlib.sha256(which.astype('<i8').tobytes()).hexdigest(),
              'upstream_identities':{s:file_sha256(root/s/'identity.json') for s in sources},
              'kernel':file_sha256(Path(__file__)),'names':names,'aliases':aliases,'score_hashes':{r['name']:r['scores_sha256'] for r in selected}}
    if (stage/'identity.json').exists() and json.loads((stage/'identity.json').read_text())!=identity:raise ValueError('correlation identity changed')
    atomic_write_json(stage/'identity.json',identity)
    sums=np.zeros((len(names),len(names)));days=np.zeros(sums.shape,np.int32);pairs=np.zeros(sums.shape,np.int64);offset=0
    checkpoint=stage/'correlation_checkpoint.json'
    if checkpoint.exists():
        saved=json.loads(checkpoint.read_text());path=stage/saved['file']
        if file_sha256(path)!=saved['sha256']:raise ValueError('correlation checkpoint changed')
        with np.load(path,allow_pickle=False) as z:sums=z['sums'];days=z['days'];pairs=z['pairs'];offset=int(z['offset'])
        if sums.shape!=(len(names),len(names)) or days.shape!=sums.shape or pairs.shape!=sums.shape or not 0<=offset<=len(which):raise ValueError('correlation checkpoint axes or offset differ')
    for position in range(offset,len(which)):
        t=which[position];mask=eligible[t]
        block=np.stack([x[t,mask] for x in matrices])
        correlation,count=exact_day_spearman(block);finite=np.isfinite(correlation)
        sums+=np.where(finite,correlation,0);days+=finite;pairs+=np.where(finite,count,0)
        if (position+1)%25==0 or position+1==len(which):
            archive=stage/f'correlation_{position+1:05d}.npz';temporary=archive.with_suffix('.tmp')
            with temporary.open('wb') as stream:np.savez_compressed(stream,sums=sums,days=days,pairs=pairs,offset=position+1)
            os.replace(temporary,archive)
            atomic_write_json(checkpoint,{'file':archive.name,'sha256':file_sha256(archive),'offset':position+1})
            status={'state':'correlating','completed_days':position+1,'total_days':len(which),'factors':len(names),'elapsed_seconds':time.monotonic()-started}
            atomic_write_json(stage/'status.json',status);print(status,flush=True)
    corr=np.divide(sums,days,out=np.full(sums.shape,np.nan),where=days>0)
    np.savez_compressed(stage/'correlations.npz',names=np.asarray(names),correlation=corr,valid_days=days,pair_observations=pairs)
    canonical={name:i for i,name in enumerate(names)}
    all_names=[r['name'] for r in positive];indexes=[canonical[aliases.get(name,name)] for name in all_names]
    full=corr[np.ix_(indexes,indexes)]
    output={}
    for h in (1,5,10):
        keep,removed=greedy_select(all_names,means[h],full)
        kept=[all_names[i] for i in keep];matrix=full[np.ix_(keep,keep)]
        triangle=np.abs(matrix[np.triu_indices(len(keep),1)]);maximum=float(np.nanmax(triangle)) if np.isfinite(triangle).any() else None
        payload={'horizon':h,'selected':[by_name[n] for n in kept],'removed':removed,'threshold':.8,'max_absolute_mean_correlation':maximum,
                 'unestimated_retained_pairs':int(np.isnan(triangle).sum()),'selection_period':protocol['train']}
        atomic_write_json(stage/f'selected_{h}d.json',payload,allow_nan=False)
        np.savez_compressed(stage/f'selected_{h}d_correlation.npz',names=np.asarray(kept),correlation=matrix)
        output[str(h)]={'selected_count':len(kept),'max_correlation':maximum}
    atomic_write_json(stage/'status.json',{'state':'complete','horizons':output,'elapsed_seconds':time.monotonic()-started})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--threads',type=int,default=8)
    a=p.parse_args();run(a.root,a.threads)
