"""One fixed MVO arm for the frozen N5 model; no training or parameter search."""
from __future__ import annotations
import argparse,gc,json,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from env.score_mvo import ScoreMVOConfig,ScoreMVOPortfolio
from env.backtest import EpisodeSession,PreparedEpisode,run_target_weight_episode
from env.action_schema import ActionSchema
from env.contracts import PortfolioExecutionSpec
from env.fees import DEFAULT_FEE_SCHEDULE
from env.forward_returns import open_forward_returns
from env.simulator import DaySimulator
from .portfolio import PreparedModelPortfolio
from .final_evaluation import write_csv
from utils.atomic_file import atomic_write_json,file_sha256


def evaluate_account(prepared,cfg,out,tag):
    rt=prepared.runtime;episode=prepared.episode;begin=time.monotonic()
    returns=open_forward_returns(rt.data,1)
    schema=ActionSchema(factor_names=prepared.factors.factor_names,filter_names=prepared.factors.filter_names,
        fixed_buy_n=rt.n_stocks,fixed_sell_m=rt.n_stocks,fixed_turnover_rate=1.,
        fixed_rebalance_band_pct=.01,schema_version='score-mvo-native-v1')
    day_config=schema.decode(np.ones(schema.action_dim))
    session=EpisodeSession(episode,action_schema=schema,initial_cash=prepared.initial_cash,
        fees=DEFAULT_FEE_SCHEDULE,portfolio_spec=PortfolioExecutionSpec(rt.n_stocks,1.,allow_cash=True))
    strategy=ScoreMVOPortfolio(rt.stock_codes,horizon=5,config=cfg)
    lookup={c:i for i,c in enumerate(rt.stock_codes)};date_lookup={str(d):i for i,d in enumerate(rt.trade_dates)}
    audits=dict(invalid_open=0,illegal_side=0,invalid_fill=0,t1_sell_violation=0)
    fills=[];daily=[];weights=[];original=DaySimulator.step
    snapshots=out/f'{tag}_qp_snapshots';snapshots.mkdir()
    selected=set(np.linspace(episode.decision_start,episode.decision_stop-2,7,dtype=int).tolist())
    def provide(context):
        i=context.current_index
        result=strategy.targets(context.current_market,context.current_account,i,prepared.prediction[i],returns[:i])
        inp=strategy.last_inputs;w=np.zeros(rt.n_stocks);w[inp['indices']]=inp['weights'];weights.append(w)
        if i in selected:np.savez_compressed(snapshots/f'{i}.npz',**inp,scores=prepared.prediction[i,inp['indices']])
        d=strategy.diagnostics[-1]
        if (i-episode.decision_start+1)%250==0:
            print(dict(stage='progress',split=tag,days=i-episode.decision_start+1,candidates=d['candidates'],
                solver_seconds=round(sum(x['seconds'] for x in strategy.diagnostics),3)),flush=True)
        return result
    def observe(self,account,plan,open_prices,next_open_prices,**kwargs):
        result=original(self,account,plan,open_prices,next_open_prices,**kwargs)
        i=date_lookup[plan.decision_date];legal=episode.market_at(i).trade_legality(day_config.limit_up_protection)
        for f in result.fills:
            j=lookup[f.code];opening=float(rt.field('open')[i,j]);buy=f.side=='buy'
            audits['invalid_open']+=int(not(np.isfinite(opening) and opening>0))
            audits['illegal_side']+=int(not(legal.buy_allowed[j] if buy else legal.sell_allowed[j]))
            audits['invalid_fill']+=int(f.quantity<=0 or not np.isfinite(f.price) or f.price<=0 or f.fee<0)
            audits['t1_sell_violation']+=int(not buy and f.quantity>account.sellable_positions.get(f.code,0))
            fills.append(dict(date=plan.decision_date,code=f.code,side=f.side,quantity=int(f.quantity),price=f.price,fee=f.fee))
        daily.append(dict(date=plan.decision_date,positions=sum(q>0 for q in result.account_state.positions.values()),
            gross_notional=float(result.diagnostics['gross_traded_notional']),fees=float(result.diagnostics['total_fees']),
            pretrade_nav=float(result.diagnostics['pretrade_nav']),**strategy.diagnostics[-1]))
        return result
    DaySimulator.step=observe
    try:trace=run_target_weight_episode(session,day_config,provide,record_details=False)
    finally:DaySimulator.step=original
    assert not any(audits.values()),audits
    assert trace.full_investment_contract_satisfied and np.min(trace.cash)>=-1e-8
    np.testing.assert_allclose(trace.portfolio_returns,trace.nav[1:]/trace.nav[:-1]-1,rtol=1e-10,atol=1e-12)
    metrics=trace.metrics.as_dict()
    metrics.update(mean_exposure=float(trace.exposure.mean()),mean_cash_fraction=float((1-trace.exposure).mean()),
        mean_gross_turnover=trace.sum_gross_turnover_ratio/len(trace.portfolio_returns),total_fees=trace.total_fees,
        max_positions=max(r['positions'] for r in daily),mean_positions=float(np.mean([r['positions'] for r in daily])))
    path=out/f'{tag}.npz';np.savez_compressed(path,nav=trace.nav,returns=trace.portfolio_returns,cash=trace.cash,
        exposure=trace.exposure,decision_dates=trace.decision_dates,next_dates=trace.next_decision_dates)
    np.savez_compressed(out/f'{tag}_targets.npz',weights=np.stack(weights),dates=trace.decision_dates)
    write_csv(out/f'{tag}_fills.csv',fills);write_csv(out/f'{tag}_daily.csv',daily)
    status={s:sum(d['status']==s for d in daily) for s in sorted(set(d['status'] for d in daily))}
    result=dict(strategy='score_mvo',split=tag,metrics=metrics,audit=audits,fees=asdict(DEFAULT_FEE_SCHEDULE),
        parameters=asdict(cfg),trace_sha256=file_sha256(path),fills_sha256=file_sha256(out/f'{tag}_fills.csv'),
        targets_sha256=file_sha256(out/f'{tag}_targets.npz'),solver_status_counts=status,
        max_candidates=max(d['candidates'] for d in daily),solver_seconds=sum(d['seconds'] for d in daily),
        funding_floor_days=sum(d['effective_turnover']>cfg.turnover_limit+1e-8 for d in daily),
        seconds=time.monotonic()-begin)
    atomic_write_json(out/f'{tag}.json',result,allow_nan=False)
    return result


def run(root:Path,parent:Path,smoke=False):
    from scripts.run_research_guarded import require_guarded_job
    require_guarded_job();started=time.monotonic();root.mkdir(parents=True,exist_ok=True)
    out=root/('smoke' if smoke else 'evaluation');out.mkdir(exist_ok=False)
    protocol=json.loads((parent/'protocol.json').read_text(encoding='utf8'));cfg=ScoreMVOConfig()
    assert asdict(DEFAULT_FEE_SCHEDULE)==protocol['fees']
    if not smoke:
        benchmark=json.loads((root/'smoke/complete.json').read_text())
        assert benchmark['state']=='complete' and benchmark['parameters']==asdict(cfg)
    sources={str(p):file_sha256(p) for d in ['env','ai/supervised'] for p in Path(d).glob('*.py')}
    frozen=dict(parent=str(parent.resolve()),horizon=5,parameters=asdict(cfg),fees=protocol['fees'],
        initial_cash=protocol['initial_cash'],model_receipt_sha256=file_sha256(parent/'models/n5/complete.json'),
        sources=sources,config_sha256=file_sha256('configs/config.json'),search_trials=0,model_training_runs=0,
        scope='same frozen N5 scores, replacement of rank-buffer allocation by ordinary MVO; no7-day hold or2-stock daily exit cap',
        signal='candidate score centered and standardized, multiplied by RMS5-day risk volatility; not calibrated expected percentage return',
        risk='last120 known economic open-to-open daily returns;50%diagonal shrinkage; covariance multiplied by5',
        constraints='long-only; full-investment optimizer;5%new/increased position cap with current drift/locked weights grandfathered;L1.1 target turnover except explicit cash funding floor; if legal cap capacity below1, hold actual weights and cash',
        candidate_policy='top50 currently buyable predictions plus every actual holding;50 is not a hard holding count',
        execution='same local fees/slippage,1%band,lot/T+1/lock/settlement; actual turnover may differ from target turnover',
        smoke=smoke)
    atomic_write_json(out/'frozen.json',frozen);results=[]
    for split in (['fit'] if smoke else ['fit','internal_validation','train','validation','test']):
        prepared=PreparedModelPortfolio(parent,5,split)
        if smoke:
            e=prepared.episode;prepared.episode=PreparedEpisode.build(prepared.runtime,prepared.factors,
                decision_start=e.decision_start,decision_stop=min(e.decision_stop,e.decision_start+253),encode_observations=False)
        r=evaluate_account(prepared,cfg,out,split);results.append(r)
        print(dict(stage='completed',split=split,days=r['metrics']['transition_count'],seconds=r['seconds'],
            solver_seconds=r['solver_seconds'],max_candidates=r['max_candidates'],audit=r['audit'],
            **({} if smoke else {'cagr':r['metrics']['annualized_return'],'calmar':r['metrics']['calmar']})),flush=True)
        del prepared;gc.collect()
    for p,sha in sources.items():assert file_sha256(p)==sha,p
    assert file_sha256('configs/config.json')==frozen['config_sha256']
    assert file_sha256(parent/'models/n5/complete.json')==frozen['model_receipt_sha256']
    write_csv(out/'portfolio_results.csv',[dict(split=r['split'],**r['metrics']) for r in results])
    atomic_write_json(out/'complete.json',{'state':'complete','parameters':asdict(cfg),'results':results,
        'seconds':time.monotonic()-started,'frozen_sha256':file_sha256(out/'frozen.json')})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--parent',type=Path,required=True);ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args();run(a.root,a.parent,a.smoke)
