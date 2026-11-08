"""Post-training native replay regression; never trains or chooses parameters."""
import gc,gzip,json,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
from offline_data import load_runtime_slice
from factor import precompute_factors
from env.backtest import PreparedEpisode,EpisodeSession,run_day_config_episode
from env.action_schema import ActionSchema
from env.fees import DEFAULT_FEE_SCHEDULE
from env.prefilter import prefilter_n_from_config
from ai.ga.config import canonicalize_individual_config
from ai.supervised.periods import observation_splits,interval,data_end
from ai.supervised.multihorizon_search import load_forecasts
from ai.supervised.portfolio import PreparedModelPortfolio
from ai.supervised.explainability import ReplayAudit,json_value
from testback.economic_positions import EconomicPositionLedger
from utils.atomic_file import atomic_write_json,file_sha256
from scripts.run_research_guarded import require_guarded_job

require_guarded_job();root=Path(__file__).resolve().parent;config=json.loads((root/'protocol.json').read_text(encoding='utf8'));winner=json.loads((root/'search/winner.json').read_text(encoding='utf8'));p=winner['winner']['parameters'];begun=time.monotonic()
for name in ['final/complete.json','static_baseline/complete.json','models/final/complete.json']:assert json.loads((root/name).read_text(encoding='utf8'))['state']=='complete'
out=root/'regression';out.mkdir(exist_ok=False);sources={str(f):file_sha256(f) for group in ['env','factor','factor_db','offline_data','ai/supervised','testback'] for f in Path(group).rglob('*.py')};sources[str(root/'search/winner.json')]=file_sha256(root/'search/winner.json');sources[str(root/'models/final/model.txt')]=file_sha256(root/'models/final/model.txt');records=[]
for f in [Path(__file__),root/'protocol.json',root/'active_data.json',Path(config['static_config_path']),root/'static_baseline/identity.json',root/'static_baseline/complete.json',root/'final/complete.json',root/'models/final/complete.json',root/'prelaunch_review.json',root/'stages/static_baseline/inputs.json',*Path('utils').rglob('*.py'),*Path('configs').rglob('*.py'),Path('ai/ga/config.py'),Path('scripts/run_research_guarded.py')]:sources[str(f)]=file_sha256(f)
atomic_write_json(out/'inputs.json',{'state':'bound_before_replay','source_sha256':sources,'model_training':False,'parameter_selection':False})

def extrema(trace):
    rows=[{'decision_date':str(d),'settlement_date':str(n),'return_pct':float(r)*100,'pretrade_nav':float(trace.nav[i]),'ending_nav':float(trace.nav[i+1])} for i,(d,n,r) in enumerate(zip(trace.decision_dates,trace.next_decision_dates,trace.portfolio_returns))]
    return {'highest':max(rows,key=lambda x:x['return_pct']),'lowest':min(rows,key=lambda x:x['return_pct']),'top_positive':sorted(rows,key=lambda x:x['return_pct'],reverse=True)[:10],'top_absolute':sorted(rows,key=lambda x:abs(x['return_pct']),reverse=True)[:10]}

def compare(strategy,split,trace):
    source=root/('final' if strategy=='ml' else 'static_baseline')/f'{split}.npz'
    with np.load(source) as prior:
        for name,new in [('nav',trace.nav),('returns',trace.portfolio_returns),('cash',trace.cash),('exposure',trace.exposure),('decision_dates',np.asarray(trace.decision_dates)),('next_dates',np.asarray(trace.next_decision_dates))]:np.testing.assert_array_equal(prior[name],new)
    return file_sha256(source)

for split in observation_splits(config):
    rt,cube,provenance=load_forecasts(root,split);scores=np.sum(cube*(np.asarray(p['weights'])/10),axis=2);del cube
    prepared=PreparedModelPortfolio.from_scores(rt,scores,initial_cash=config['initial_cash'],source_hash=file_sha256(root/'search/winner.json'),split=split)
    target=out/f'ml_{split}';target.mkdir();audit=ReplayAudit(target,prepared,p);ledger=EconomicPositionLedger()
    def observe(before,transition):audit(before,transition);ledger(before,transition)
    try:trace,m,_=prepared.evaluate('rank_buffer',{k:p[k] for k in ['buy_n','sell_m','hold_thresh','max_replacements']},details=True,on_transition=observe);audit.finish()
    finally:audit.stream.close()
    sha=compare('ml',split,trace);summary=ledger.summary(prepared.final_account)
    best=max(audit.days,key=lambda d:d['return_pct']);worst=min(audit.days,key=lambda d:d['return_pct'])
    atomic_write_json(target/'extreme_days.json',json_value({'highest':best,'lowest':worst,'top_positive':sorted(audit.days,key=lambda d:d['return_pct'],reverse=True)[:10],'top_absolute':sorted(audit.days,key=lambda d:abs(d['return_pct']),reverse=True)[:10]}))
    records.append({'strategy':'ml','split':split,'prior_trace_sha256':sha,'all_six_trace_arrays_exact':True,'metrics':m,'ledger':summary,'trade_audit':audit.counts,'highest_return_pct':best['return_pct'],'worst_return_pct':worst['return_pct'],'source_models':provenance});print({'regression':'ml','split':split,'exact':True},flush=True)
    del trace,prepared,audit,ledger,rt,scores;gc.collect()

active=json.loads((root/'active_data.json').read_text(encoding='utf8'));rt=load_runtime_slice(active['runtime_path'],config['train'][0],data_end(config),preload_rows=100001,cache_dir=Path(config['feature_runtime_cache']));factors=precompute_factors(rt);payload=json.loads(Path(config['static_config_path']).read_text(encoding='utf8'));schema=ActionSchema.from_config(payload,factor_names=factors.factor_names,filter_names=factors.filter_names);canonical,day=canonicalize_individual_config(payload,action_schema=schema);assert asdict(DEFAULT_FEE_SCHEDULE)==config['fees']
for split in observation_splits(config):
    bounds=interval(config,split);lo=int(np.searchsorted(rt.trade_dates,np.datetime64(bounds[0])));hi=int(np.searchsorted(rt.trade_dates,np.datetime64(bounds[1]),side='right'))
    episode=PreparedEpisode.build(rt,factors,decision_start=lo,decision_stop=hi,prefilter_n=prefilter_n_from_config(payload),encode_observations=False,action_schema=schema);session=EpisodeSession(episode,action_schema=schema,initial_cash=config['initial_cash'],fees=DEFAULT_FEE_SCHEDULE);target=out/f'static_{split}';target.mkdir();ledger=EconomicPositionLedger();stream=gzip.open(target/'native_steps.jsonl.gz','wt',encoding='utf8');counts={'invalid_open':0,'illegal_side':0,'fee_error':0}
    def observe_static(before,transition):
        ledger(before,transition);s=transition.step_result;plan=transition.order_plan;market=episode.market_at(rt.index_of(plan.decision_date));legal=market.trade_legality(plan.day_config.limit_up_protection)
        for fill in s.fills:
            j=episode.code_to_index[fill.code];counts['invalid_open']+=int(not np.isfinite(market.open_prices[j]) or fill.price!=market.open_prices[j]);counts['illegal_side']+=int(not(legal.buy_allowed[j] if fill.side=='buy' else legal.sell_allowed[j]))
            expected=DEFAULT_FEE_SCHEDULE.buy_fee(fill.quantity*fill.price) if fill.side=='buy' else DEFAULT_FEE_SCHEDULE.sell_fee(fill.quantity*fill.price)
            counts['fee_error']+=int(not np.isclose(fill.fee,expected,rtol=1e-10,atol=1e-8))
        stream.write(json.dumps(json_value({'before':before,'after':s.account_state,'plan':plan,'fills':s.fills,'diagnostics':s.diagnostics,'info':transition.info}),ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    try:trace=run_day_config_episode(session,lambda _:day,record_details=True,on_transition=observe_static)
    finally:stream.close()
    assert not any(counts.values()),counts;sha=compare('static',split,trace);extreme=extrema(trace);extreme['native_steps_sha256']=file_sha256(target/'native_steps.jsonl.gz');atomic_write_json(target/'extreme_days.json',extreme)
    records.append({'strategy':'static','split':split,'prior_trace_sha256':sha,'all_six_trace_arrays_exact':True,'ledger':ledger.summary(session.current_account),'trade_audit':counts,'metrics':trace.metrics.as_dict(),'highest_return_pct':extreme['highest']['return_pct'],'worst_return_pct':extreme['lowest']['return_pct'],'native_steps_sha256':extreme['native_steps_sha256']});print({'regression':'static','split':split,'exact':True},flush=True)
    del trace,session,episode,ledger;gc.collect()
for path,sha in sources.items():assert file_sha256(path)==sha,path
atomic_write_json(out/'complete.json',{'state':'complete','results':records,'source_sha256':sources,'seconds':time.monotonic()-begun,'training_runs':0,'parameter_selection':False,'scope':'Actual native replay and observer regression; all source validity/auction capacity boundaries remain. Extreme-return source review is separate.'})
