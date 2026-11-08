"""Fit-only selection at seven fold cutoffs plus final refit cutoff."""
import argparse,json,time
from pathlib import Path
import numpy as np
from numba import set_num_threads
from env.forward_returns import label_split_mask
from .selection import exact_day_spearman
from .multihorizon_admission import verify_data
from utils.atomic_file import atomic_write_json,file_sha256

def fold_specs(config):
    return [dict(name=f'fold{y}',fit=['2005-01-01',f'{y-2}-12-31'],stop=[f'{y-1}-01-01',f'{y-1}-12-31'],evaluate=[f'{y}-01-01',f'{y}-12-31']) for y in config['fold_years']]+[dict(name='final',fit=config['train'])]

def select_one(names,means,corr,days,threshold=.8,minimum_days=60):
    order=sorted([i for i,n in enumerate(names) if means.get(n) is not None and means[n]>0],key=lambda i:(-means[names[i]],names[i]))
    kept=[];removed=[]
    for i in order:
        conflict=next((j for j in kept if days[i,j]<minimum_days or not np.isfinite(corr[i,j]) or abs(corr[i,j])>=threshold),None)
        if conflict is None:kept.append(i)
        else:removed.append({'name':names[i],'against':names[conflict],'reason':'insufficient_common_days' if days[i,conflict]<minimum_days or not np.isfinite(corr[i,conflict]) else 'high_correlation','correlation':float(corr[i,conflict]) if np.isfinite(corr[i,conflict]) else None})
    return kept,removed

def run(root):
    from scripts.run_research_guarded import require_guarded_job
    require_guarded_job();set_num_threads(8);start=time.monotonic()
    config=json.loads((root/'protocol.json').read_text(encoding='utf8'));stage=root/'fold_selection';stage.mkdir(exist_ok=True)
    source=root/'features';assert json.loads((source/'status.json').read_text())['state']=='complete'
    data_identity=verify_data(root)
    inventory=json.loads((source/'inventory.json').read_text(encoding='utf8'));dates=np.load(source/'dates.npy');eligible=np.load(source/'label_eligible.npy',mmap_mode='r')
    specs=fold_specs(config);means={};counts={};ends={};positive=set()
    for spec in specs:
        mask=label_split_mask(dates,*spec['fit'],60);ends[spec['name']]=int(np.flatnonzero(mask)[-1]);means[spec['name']]={};counts[spec['name']]={}
        for h in config['model_horizons']:
            mh={};ch={}
            for name,r in inventory.items():
                p=Path(r['daily_ic'][str(h)]['path']);assert file_sha256(p)==r['daily_ic'][str(h)]['sha256']
                x=np.load(p)[mask,1];x=x[np.isfinite(x)];ch[name]=len(x)
                mh[name]=float(x.mean()) if len(x)>=60 else None
                if mh[name] is not None and mh[name]>0:positive.add(name)
            means[spec['name']][h]=mh;counts[spec['name']][h]=ch
    # Numeric aliases share the same canonical correlation row, without dropping names silently.
    canonical={};aliases={}
    for n in sorted(positive):
        sha=inventory[n]['sha256'];canonical.setdefault(sha,n);aliases[n]=canonical[sha]
    names=sorted(set(aliases.values()));index={n:i for i,n in enumerate(names)}
    arrays=[]
    for n in names:
        r=inventory[n];assert file_sha256(r['path'])==r['sha256'];arrays.append(np.load(r['path'],mmap_mode='r'))
    digest={'protocol':file_sha256(root/'protocol.json'),'data_manifest':data_identity,'inventory':file_sha256(source/'inventory.json'),'kernel':file_sha256('ai/supervised/selection.py'),'driver':file_sha256(__file__),'names':names,'endpoints':ends}
    if (stage/'identity.json').exists():assert json.loads((stage/'identity.json').read_text())==digest
    atomic_write_json(stage/'identity.json',digest)
    n=len(names);sums=np.zeros((n,n));days=np.zeros((n,n),np.int32);first=0
    checkpoint=stage/'checkpoint.json'
    if checkpoint.exists():
        cp=json.loads(checkpoint.read_text());assert file_sha256(stage/cp['file'])==cp['sha256']
        z=np.load(stage/cp['file']);sums=z['sums'];days=z['days'];first=int(z['next'])
    for t in range(first,max(ends.values())+1):
        values=np.stack([a[t,eligible[t]] for a in arrays]);corr,_=exact_day_spearman(values);good=np.isfinite(corr)
        sums+=np.where(good,corr,0);days+=good
        if t in ends.values():
            matrix=np.divide(sums,days,out=np.full(sums.shape,np.nan),where=days>0)
            for spec in [s for s in specs if ends[s['name']]==t]:
                directory=stage/spec['name'];directory.mkdir(exist_ok=True);union=set();horizons={}
                allnames=sorted(positive);ii=[index[aliases[x]] for x in allnames];expanded=matrix[np.ix_(ii,ii)];valid=days[np.ix_(ii,ii)].copy()
                # Exact identical columns have known correlation1 even if daily variance is absent.
                for a in range(len(allnames)):
                    for b in range(a):
                        if ii[a]==ii[b]:expanded[a,b]=expanded[b,a]=1.;valid[a,b]=valid[b,a]=10**9
                for h in config['model_horizons']:
                    keep,removed=select_one(allnames,means[spec['name']][h],expanded,valid)
                    retained=[allnames[i] for i in keep];union.update(retained)
                    payload={'horizon':h,'fit':spec['fit'],'last_feature_date':str(dates[t]),'label_end_boundary':spec['fit'][1],
                        'selected':[{'name':x,'rank_ic':means[spec['name']][h][x],'valid_days':counts[spec['name']][h][x]} for x in retained],
                        'removed':removed,'threshold':.8}
                    atomic_write_json(directory/f'selected_{h}d.json',payload);horizons[h]=len(retained)
                ordered=sorted(union);idx=[index[aliases[x]] for x in ordered]
                np.savez_compressed(directory/'correlation.npz',names=np.asarray(ordered),correlation=matrix[np.ix_(idx,idx)],valid_days=days[np.ix_(idx,idx)])
                atomic_write_json(directory/'complete.json',{'state':'complete','spec':spec,'features':ordered,'per_horizon_counts':horizons,'fit_only':True})
                print({'fold':spec['name'],'features':len(ordered),'counts':horizons},flush=True)
        if (t+1)%25==0 or t==max(ends.values()):
            file=f'checkpoint_{t+1}.npz';np.savez_compressed(stage/file,sums=sums,days=days,next=t+1)
            atomic_write_json(checkpoint,{'file':file,'sha256':file_sha256(stage/file)})
            atomic_write_json(stage/'status.json',{'state':'correlating','days':t+1,'total':max(ends.values())+1,'factors':n,'seconds':time.monotonic()-start})
    assert verify_data(root)==data_identity
    atomic_write_json(stage/'status.json',{'state':'complete','seconds':time.monotonic()-start,'folds':len(specs)})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
