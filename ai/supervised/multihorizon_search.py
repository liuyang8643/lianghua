"""Deterministic discrete GA on a continuous, year-switching OOF account."""
import argparse,gc,hashlib,itertools,json,random,time
from pathlib import Path
import numpy as np
from dataclasses import asdict
from env.fees import DEFAULT_FEE_SCHEDULE
from offline_data import load_runtime_slice
from .feature_data import rank_features
from .periods import oof_interval
from .portfolio import PreparedModelPortfolio
from .multihorizon_admission import verify_data
from .final_evaluation import write_csv
from utils.atomic_file import atomic_write_json,file_sha256

def random_weights(rng):
    cuts=[-1,*sorted(rng.sample(range(14),4)),14]
    return [cuts[i+1]-cuts[i]-1 for i in range(5)]

def candidate_key(p):return json.dumps(p,sort_keys=True,separators=(',',':'))
def fitness(r):
    m=r['metrics'];positive=m['annualized_return']>0
    p=r.get('parameters');tie=tuple(-v for v in [*p['weights'],p['buy_n'],p['sell_m'],p['hold_thresh'],p['max_replacements']]) if p else ()
    return (int(positive),m['calmar'] if positive else m['annualized_return'],-m['mean_gross_turnover'],tie)

def random_candidate(space,rng):
    return {'weights':random_weights(rng),**{k:rng.choice(space[k]) for k in ['buy_n','sell_m','hold_thresh','max_replacements']}}

def child_candidate(space,population,records,rng):
    def parent():return max(rng.sample(population,min(3,len(population))),key=lambda p:fitness(records[candidate_key(p)]))
    a,b=parent(),parent();child={'weights':list(rng.choice([a['weights'],b['weights']]))}
    for k in ['buy_n','sell_m','hold_thresh','max_replacements']:child[k]=rng.choice([a[k],b[k]]) if rng.random()>.2 else rng.choice(space[k])
    if rng.random()<.6:
        for _ in range(rng.randint(1,3)):
            source=rng.choice([i for i,v in enumerate(child['weights']) if v]);target=rng.randrange(5)
            child['weights'][source]-=1;child['weights'][target]+=1
    return child

def load_forecasts(root,split):
    c=json.loads((root/'protocol.json').read_text(encoding='utf8'));a=json.loads((root/'active_data.json').read_text())
    assert file_sha256(a['runtime_path'])==a['runtime_sha256']
    interval=oof_interval(c) if split=='oof' else c[split]
    rt=load_runtime_slice(a['runtime_path'],*interval,preload_rows=252,cache_dir=Path(c['portfolio_runtime_cache']))
    dates=np.load(root/'features/dates.npy');cube=np.full((rt.n_dates,rt.n_stocks,5),np.nan)
    np.testing.assert_array_equal(np.load(root/'features/codes.npy').astype(str),rt.stock_codes)
    provenance={}
    folders=[root/f'models/fold{y}' for y in c['fold_years']] if split=='oof' else [root/'models/final']
    for folder in folders:
        done=json.loads((folder/'complete.json').read_text());assert done['state']=='complete'
        for filename,key in [('predictions.npy','predictions_sha256'),('rows.npy','rows_sha256'),('model.txt','model_sha256')]:assert file_sha256(folder/filename)==done[key]
        rows=np.load(folder/'rows.npy');dd=dates[rows];pos=np.searchsorted(rt.trade_dates,dd);valid=pos<len(rt.trade_dates);valid[valid]&=rt.trade_dates[pos[valid]]==dd[valid]
        cube[pos[valid]]=np.load(folder/'predictions.npy',mmap_mode='r')[valid];provenance[folder.name]=file_sha256(folder/'complete.json')
    at=np.searchsorted(dates,rt.trade_dates);valid=at<len(dates);valid[valid]&=dates[at[valid]]==rt.trade_dates[valid]
    eligible=np.zeros(cube.shape[:2],bool);eligible[valid]=np.load(root/'features/eligible.npy',mmap_mode='r')[at[valid]]
    pool=eligible&np.isfinite(cube).all(axis=2)
    assert pool[rt.decision_start:rt.decision_stop].any(axis=1).all(),'OOF prediction calendar has a gap'
    for k in range(5):cube[:,:,k]=rank_features(cube[:,:,k],pool)
    return rt,cube,provenance

def evaluate_candidate(rt,cube,p,initial_cash,source_hash,split,details=False,on_prepared=None,prepared_cache=None):
    weights=np.asarray(p['weights'],float)/10
    assert len(weights)==5 and np.all(weights>=0) and sum(p['weights'])==10
    cache_key=(id(rt),tuple(p['weights']))
    if prepared_cache is not None and cache_key in prepared_cache:
        prepared=prepared_cache.pop(cache_key);prepared_cache[cache_key]=prepared
    else:
        scores=np.sum(cube*weights,axis=2)
        prepared=PreparedModelPortfolio.from_scores(rt,scores,initial_cash=initial_cash,source_hash=source_hash,split=split)
        if prepared_cache is not None:
            prepared_cache[cache_key]=prepared
            while len(prepared_cache)>2:prepared_cache.pop(next(iter(prepared_cache)))
    if on_prepared is not None:on_prepared(prepared)
    parameters={k:p[k] for k in ['buy_n','sell_m','hold_thresh','max_replacements']}
    return prepared.evaluate('rank_buffer',parameters,details=details)

def run(root):
    from scripts.run_research_guarded import require_guarded_job
    require_guarded_job();start=time.monotonic();c=json.loads((root/'protocol.json').read_text(encoding='utf8'));space=c['ga']
    data_identity=verify_data(root)
    assert asdict(DEFAULT_FEE_SCHEDULE)==c['fees']
    assert json.loads((root/'models/oof_complete.json').read_text())['state']=='complete'
    stage=root/'search';stage.mkdir(exist_ok=True)
    if (stage/'winner.json').exists():raise RuntimeError('search already sealed; do not tune after observations')
    rt,cube,provenance=load_forecasts(root,'oof')
    identity={'protocol':file_sha256(root/'protocol.json'),'data_manifest':data_identity,'models':provenance,'source':file_sha256(__file__),
        'tie_break':'lower turnover, then ascending weights[3,7,15,30,60],buy_n,sell_m,hold_thresh,max_replacements',
        'portfolio_source':file_sha256('ai/supervised/portfolio.py'),'strategy_source':file_sha256('env/qlib_strategy.py'),
        'search_interval':oof_interval(c),'account_resets':'only once per candidate; no resets at fold switches'}
    if (stage/'identity.json').exists():assert json.loads((stage/'identity.json').read_text())==identity
    atomic_write_json(stage/'identity.json',identity);records={};(stage/'trials').mkdir(exist_ok=True)
    for p in (stage/'trials').glob('*.json'):
        r=json.loads(p.read_text());assert file_sha256(p.with_suffix('.npz'))==r['trace_sha256'];records[candidate_key(r['parameters'])]=r
    rng=random.Random(space['seed']);population=[];prepared_cache={}
    for k in range(5):
        w=[0]*5;w[k]=10;population.append(dict(weights=w,buy_n=20,sell_m=60,hold_thresh=7,max_replacements=2))
    population.append(dict(weights=[2]*5,buy_n=20,sell_m=60,hold_thresh=7,max_replacements=2))
    while len(population)<space['population']:
        p=random_candidate(space,rng)
        if p not in population:population.append(p)
    start_gen=0;stale=0;anchor=None
    checkpoint=stage/'checkpoint.json'
    if checkpoint.exists():
        x=json.loads(checkpoint.read_text());population=x['population'];start_gen=x['next_generation'];stale=x['stale'];anchor=x['anchor']
        def tuplify(v):return tuple(map(tuplify,v)) if isinstance(v,list) else v
        rng.setstate(tuplify(x['rng']))
    for generation in range(start_gen,space['generations']):
        for p in population:
            key=candidate_key(p)
            if key in records:continue
            began=time.monotonic();sha=hashlib.sha256(key.encode()).hexdigest()
            signal_hash=hashlib.sha256((file_sha256(stage/'identity.json')+str(p['weights'])).encode()).hexdigest()
            trace,metrics,_=evaluate_candidate(rt,cube,p,c['initial_cash'],signal_hash,'oof',prepared_cache=prepared_cache)
            assert metrics['allocation_contract_satisfied']
            np.testing.assert_allclose(trace.portfolio_returns,trace.nav[1:]/trace.nav[:-1]-1,rtol=1e-10,atol=1e-12)
            path=stage/'trials'/sha;np.savez_compressed(path.with_suffix('.npz'),nav=trace.nav,returns=trace.portfolio_returns,exposure=trace.exposure,cash=trace.cash,decision_dates=trace.decision_dates,next_dates=trace.next_decision_dates)
            result={'state':'complete','parameters':p,'metrics':metrics,'generation_first_seen':generation+1,'seconds':time.monotonic()-began,'trace_sha256':file_sha256(path.with_suffix('.npz'))}
            atomic_write_json(path.with_suffix('.json'),result);records[key]=result;del trace;gc.collect()
        best=max(records.values(),key=fitness);score=fitness(best)
        improvement=anchor is None or score[0]>anchor[0] or (score[0]==anchor[0] and score[1]>anchor[1]+space['min_improvement'])
        if improvement:anchor=list(score[:2]);stale=0
        else:stale+=1
        atomic_write_json(stage/f'generation_{generation+1:03d}.json',{'generation':generation+1,'population':population,'best':best,'stale':stale,'unique_trials':len(records),'rng':rng.getstate()})
        print({'generation':generation+1,'trials':len(records),'best_calmar':best['metrics']['calmar'],'stale':stale},flush=True)
        if stale>=space['patience'] or generation+1==space['generations']:break
        ranked=sorted(population,key=lambda p:fitness(records[candidate_key(p)]),reverse=True);new=ranked[:4]
        while len(new)<space['population']:
            p=random_candidate(space,rng) if rng.random()<.15 else child_candidate(space,population,records,rng)
            if p not in new:new.append(p)
        population=new
        atomic_write_json(checkpoint,{'next_generation':generation+1,'population':population,'rng':rng.getstate(),'stale':stale,'anchor':anchor})
    winner=max(records.values(),key=fitness)
    assert verify_data(root)==data_identity
    for name,sha in provenance.items():
        folder=root/'models'/name;assert file_sha256(folder/'complete.json')==sha
        done=json.loads((folder/'complete.json').read_text())
        assert file_sha256(folder/'predictions.npy')==done['predictions_sha256']
        assert file_sha256(folder/'model.txt')==done['model_sha256']
    atomic_write_json(stage/'winner.json',{'state':'qualified' if winner['metrics']['annualized_return']>0 else 'no_qualified_candidate',
        'winner':winner,'generations':generation+1,'unique_trials':len(records),'stop_reason':'patience' if stale>=space['patience'] else 'generation_limit','seconds':time.monotonic()-start,'identity_sha256':file_sha256(stage/'identity.json')})
    write_csv(stage/'trials.csv',[{'weights':r['parameters']['weights'],**{k:v for k,v in r['parameters'].items() if k!='weights'},**{k:v for k,v in r['metrics'].items() if not isinstance(v,dict)}} for r in sorted(records.values(),key=fitness,reverse=True)])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
