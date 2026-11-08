import gc,itertools,json,time,gzip,traceback
from pathlib import Path
import numpy as np
from ai.supervised.multihorizon_search import load_forecasts
from ai.supervised.portfolio import PreparedModelPortfolio
from ai.supervised.explainability import json_value
from env.forward_returns import open_forward_returns
from env.backtest import PreparedEpisode
from testback.economic_positions import EconomicPositionLedger
from utils.atomic_file import atomic_write_json,file_sha256
from scripts.run_research_guarded import require_guarded_job
from mvo_adapter import evaluate

R=Path(__file__).resolve().parent;P=R.with_name('multihorizon_pre_live_20261108_volume_fix')
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def save_trace(path,t):np.savez_compressed(path,nav=t.nav,returns=t.portfolio_returns,cash=t.cash,exposure=t.exposure,decision_dates=t.decision_dates,next_dates=t.next_decision_dates)
def prepare(split):
    rt,cube,provenance=load_forecasts(P,split);scores=np.sum(cube*np.array([.2,.3,.1,.3,.1]),axis=2);del cube;gc.collect()
    p=PreparedModelPortfolio.from_scores(rt,scores,initial_cash=read(P/'protocol.json')['initial_cash'],source_hash=file_sha256(P/'search/winner.json'),split=split)
    observed=open_forward_returns(rt.data,1)
    return p,observed,provenance
def annual(trace):
    rows=[];dates=np.asarray(trace.decision_dates).astype('datetime64[Y]').astype(str)
    for year in np.unique(dates):
        rr=trace.portfolio_returns[dates==year];nav=np.r_[1.,np.cumprod(1+rr)];dd=float(np.max(1-nav/np.maximum.accumulate(nav)));ret=float(nav[-1]-1)
        rows.append({'year':str(year),'return':ret,'drawdown':dd,'return_over_drawdown':ret/dd if dd else None})
    return rows
def main():
    require_guarded_job();start=time.monotonic();protocol=read(R/'protocol.json');assert not (R/'trials').exists();(R/'trials').mkdir()
    source={str(p):file_sha256(p) for folder in ['env','offline_data','ai/supervised','factor','factor_db','utils','testback'] for p in Path(folder).rglob('*.py')}
    for p in [R/'run.py',R/'mvo_adapter.py',R/'protocol.json',P/'search/winner.json',P/'completion_receipt.json']:source[str(p)]=file_sha256(p)
    atomic_write_json(R/'inputs.json',{'sources':source,'training':False,'external_access_before_winner':False})
    prepared,observed,provenance=prepare('oof');original=prepared.episode;rt=prepared.runtime
    atomic_write_json(R/'forecast_identity.json',provenance)
    # Benchmark is first100 chronological OOF days; selection of grid depends on walltime only.
    prepared.episode=PreparedEpisode.build(rt,prepared.factors,decision_start=rt.decision_start,decision_stop=min(rt.decision_stop,rt.decision_start+101),encode_observations=False)
    benchmark={'risk_aversion':50.,'turnover_limit':.1,'max_weight':.05,'pool_size':160};b=time.monotonic()
    t,m,diag,_=evaluate(prepared,benchmark,observed);elapsed=time.monotonic()-b;days=len(t.portfolio_returns)
    save_trace(R/'benchmark.npz',t);atomic_write_json(R/'benchmark.json',{'seconds':elapsed,'days':days,'parameters':benchmark,'metrics_not_used_for_grid':m,'diagnostics':diag})
    prepared.episode=original;estimate=elapsed/days*(rt.decision_stop-rt.decision_start-1)
    pools=[80,160] if estimate*24<4200 else [80]
    grid=[dict(zip(['risk_aversion','turnover_limit','max_weight','pool_size'],v)) for v in itertools.product([10.,50.,200.],[.05,.1],[.025,.05],pools)]
    atomic_write_json(R/'search_plan.json',{'grid':grid,'benchmark_seconds':elapsed,'estimated_seconds_per_trial':estimate,'budget_basis':'walltime only','selected_before_full_trial_results':True})
    print({'stage':'benchmark','seconds':elapsed,'estimated_per_trial':estimate,'grid':len(grid)},flush=True)
    records=[]
    for i,p in enumerate(grid):
        if (R/'STOP').exists():raise RuntimeError('STOP requested')
        # Preserve full candidate evaluation; reserve observations/report budget rather than truncate market history.
        if time.monotonic()-start>6000:break
        b=time.monotonic();stem=R/'trials'/f'{i:02d}'
        try:
            t,m,diag,_=evaluate(prepared,p,observed);save_trace(stem.with_suffix('.npz'),t)
            record={'state':'complete','parameters':p,'metrics':m,'annual':annual(t),'trace_sha256':file_sha256(stem.with_suffix('.npz')),'seconds':time.monotonic()-b,'id':i}
            with gzip.open(stem.with_suffix('.diagnostics.json.gz'),'wt',encoding='utf8') as f:json.dump(diag,f)
        except (ValueError,RuntimeError) as e:
            record={'state':'invalid','parameters':p,'error':str(e),'traceback':traceback.format_exc(),'seconds':time.monotonic()-b,'id':i}
        atomic_write_json(stem.with_suffix('.json'),record);records.append(record);print({'trial':i,**{k:record[k] for k in ['state','seconds']},'calmar':record.get('metrics',{}).get('calmar')},flush=True)
    valid=[r for r in records if r['state']=='complete'];assert valid,'No valid candidate'
    def key(r):
        m=r['metrics'];return (m['annualized_return']>0,m['calmar'] if m['annualized_return']>0 else m['annualized_return'],-m['mean_gross_turnover'],-r['id'])
    winner=max(valid,key=key);atomic_write_json(R/'winner.json',{'winner':winner,'trials':len(records),'planned':len(grid),'complete_grid':len(records)==len(grid),'selection':'training continuous OOF only','external_seen':False})
    del prepared,observed,t;gc.collect()
    # Only after immutable winner is written do final-model external observations begin.
    prepared,observed,external_provenance=prepare('validation');ledger=EconomicPositionLedger();stream=gzip.open(R/'external_steps.jsonl.gz','wt',encoding='utf8')
    def observe(before,tr):
        ledger(before,tr);s=tr.step_result;stream.write(json.dumps(json_value({'before':before,'after':s.account_state,'plan':tr.order_plan,'fills':s.fills,'diagnostics':s.diagnostics,'info':tr.info}),ensure_ascii=False)+'\n')
    try:t,m,diag,account=evaluate(prepared,winner['parameters'],observed,details=True,on_transition=observe)
    finally:stream.close()
    save_trace(R/'external.npz',t);atomic_write_json(R/'external.json',{'metrics':m,'annual':annual(t),'ledger':ledger.summary(account),'parameters':winner['parameters'],'diagnostics':diag,'models':external_provenance})
    for p,s in source.items():assert file_sha256(p)==s,p
    atomic_write_json(R/'complete.json',{'state':'complete','seconds':time.monotonic()-start,'trials':len(records),'valid':len(valid),'planned':len(grid),'winner_sha256':file_sha256(R/'winner.json'),'external_sha256':file_sha256(R/'external.npz'),'source_unchanged':True,'no_training':True})
    print({'state':'complete','seconds':time.monotonic()-start},flush=True)
if __name__=='__main__':main()
