"""Fixed three-horizon ATrade allocation on the single local account engine."""
from __future__ import annotations
import argparse,gc,json,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from offline_data import load_runtime_slice
from env.atrade_strategy import ATradeConfig,ATradeBookPortfolio,completed_volatility,completed_liquidity_pool,dispersion_exposure
from env.action_schema import ActionSchema
from env.backtest import EpisodeSession,run_target_weight_episode
from env.contracts import PortfolioExecutionSpec
from env.fees import DEFAULT_FEE_SCHEDULE
from env.simulator import DaySimulator
from .portfolio import PreparedModelPortfolio
from .final_evaluation import write_csv
from utils.atomic_file import atomic_write_json,file_sha256


def load_signal_history(root,model_sources,config):
    """Compute the chained20+250 windows once before slicing any account split."""
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf8'))
    active=json.loads((root/'active_data.json').read_text())
    assert file_sha256(active['runtime_path'])==active['runtime_sha256']
    rt=load_runtime_slice(active['runtime_path'],protocol['train'][0],protocol['test'][1],
        preload_rows=252,cache_dir=Path(protocol['portfolio_runtime_cache']))
    a,b=rt.decision_start,rt.decision_stop;dates=rt.trade_dates[a:b]
    member=(rt.field('listing_age')>=0)&~rt.field('delisted_mask')
    liquidity=completed_liquidity_pool(rt.field('amount'),member,config.liquidity_top_n)[a:b]
    volatility=completed_volatility(rt.field('close'),rt.field('preClose'),member)[a:b]
    predictions=[]
    for h in [5,20,60]:
        source=model_sources[h];folder=source/f'models/n{h}'
        done=json.loads((folder/'complete.json').read_text())
        assert file_sha256(folder/'predictions.npy')==done['predictions_sha256']
        np.testing.assert_array_equal(np.load(source/'features/dates.npy'),dates)
        np.testing.assert_array_equal(np.load(source/'features/codes.npy').astype(str),rt.stock_codes)
        predictions.append(np.where(liquidity,np.load(folder/'predictions.npy',mmap_mode='r'),np.nan))
    dispersion,percentile,scale=dispersion_exposure(predictions,config)
    return dict(dates=dates,liquidity=liquidity,volatility=volatility,dispersion=dispersion,percentile=percentile,scale=scale)


def run_accounts(episodes,predictions,volatility,scale,initial_cash,config=ATradeConfig()):
    """Three independently executed shadow books, plus merged/timed accounts.

    Every fill/settlement uses EpisodeSession. Target schedules, not realized
    NAV curves or prediction scores, are combined. No post-split information
    is used to generate a target.
    """
    first=episodes[5];rt=first.runtime;start=first.decision_start;stop=first.decision_stop-1
    rebalance_days=set(range(start,stop,config.rebalance));results={};schedules={}
    original=DaySimulator.step
    for name,horizon in [('book5',5),('book20',20),('book60',60),('anchor',None),('timed',None)]:
        episode=episodes[horizon or 5]
        schema=ActionSchema(factor_names=episode.factors.factor_names,filter_names=episode.factors.filter_names,
            fixed_buy_n=rt.n_stocks,fixed_sell_m=rt.n_stocks,fixed_turnover_rate=1.,
            fixed_rebalance_band_pct=.01,schema_version='atrade-native-allocation-v1')
        day_config=schema.decode(np.ones(schema.action_dim))
        session=EpisodeSession(episode,action_schema=schema,initial_cash=initial_cash,
            fees=DEFAULT_FEE_SCHEDULE,portfolio_spec=PortfolioExecutionSpec(rt.n_stocks,1.,allow_cash=True))
        strategy=ATradeBookPortfolio(rt.stock_codes,config);targets={};fills=[];days=[]
        lookup={c:i for i,c in enumerate(rt.stock_codes)};date_lookup={str(d):i for i,d in enumerate(rt.trade_dates)}
        audit={'illegal_side':0,'invalid_open':0,'nonrebalance_fills':0,'invalid_fill':0}
        def provide(context):
            i=context.current_index;is_rebalance=i in rebalance_days
            desired=None
            if horizon is None and is_rebalance:
                desired=(schedules[5][i]+schedules[20][i]+schedules[60][i])/3
                if name=='timed':desired=desired*scale[i]
            output=strategy.targets(context.current_market,context.current_account,i,
                predictions[horizon or 5][i],volatility[i],rebalance=is_rebalance,desired=desired)
            if is_rebalance:targets[i]=strategy.last_desired.copy()
            return output
        def observe(self,account,plan,open_prices,next_open_prices,**kwargs):
            result=original(self,account,plan,open_prices,next_open_prices,**kwargs)
            i=date_lookup[plan.decision_date];legal=episode.market_at(i).trade_legality(day_config.limit_up_protection)
            audit['nonrebalance_fills']+=int(i not in rebalance_days and bool(result.fills))
            for f in result.fills:
                j=lookup[f.code];price=rt.field('open')[i,j]
                audit['invalid_open']+=int(not(np.isfinite(price) and price>0))
                audit['illegal_side']+=int(not(legal.buy_allowed[j] if f.side=='buy' else legal.sell_allowed[j]))
                audit['invalid_fill']+=int(f.quantity<=0 or not np.isfinite(f.price) or f.price<=0 or f.fee<0)
                fills.append(dict(date=plan.decision_date,code=f.code,side=f.side,quantity=int(f.quantity),price=f.price,fee=f.fee))
            days.append(dict(date=plan.decision_date,rebalance=i in rebalance_days,
                positions=sum(q>0 for q in result.account_state.positions.values()),fills=len(result.fills),
                desired_exposure=float(targets[i].sum()) if i in targets else None,
                scale=float(scale[i]) if name=='timed' and i in rebalance_days else None))
            return result
        DaySimulator.step=observe
        try:trace=run_target_weight_episode(session,day_config,provide,record_details=False)
        finally:DaySimulator.step=original
        assert not any(audit.values()),audit
        assert trace.full_investment_contract_satisfied
        if horizon:schedules[horizon]=targets
        results[name]=dict(trace=trace,audit=audit,fills=fills,daily=days,targets=targets,
                           allocation_diagnostics=strategy.diagnostics)
    return results


def evaluate(root:Path):
    from scripts.run_research_guarded import require_guarded_job
    require_guarded_job();started=time.monotonic();out=root/'evaluation';out.mkdir(exist_ok=False)
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf8'))
    model_sources={int(h):Path(p) for h,p in json.loads((root/'model_sources.json').read_text()).items()}
    config=ATradeConfig()
    sources={str(p):file_sha256(p) for d in ('env','ai/supervised') for p in Path(d).glob('*.py')}
    frozen={'strategy':asdict(config),'horizons':[5,20,60],'model_sources':{str(k):str(v) for k,v in model_sources.items()},
        'model_receipts':{str(h):file_sha256(p/f'models/n{h}/complete.json') for h,p in model_sources.items()},
        'fees':asdict(DEFAULT_FEE_SCHEDULE),'initial_cash':protocol['initial_cash'],'search_trials':0,
        'sources':sources,'config_sha256':file_sha256('configs/config.json'),
        'signal_history':'Compute completed liquidity/volatility and rolling dispersion over full2005-2026 prediction history once, then slice; never reset chained windows at split boundaries.',
        'adaptations':['Local frozen262-feature open-time predictors, only N60 newly trained; no rolling retraining or h5 head reweighting.',
            'Apply upstream liquidity top500 to predictions at allocation time using T-1 completed20-day amount; model training universe remains local broad PIT.',
            'Volatility uses completed official close/preClose returns, not T HLC or upstream forward-adjusted close.',
            'Retain native fees, exchange lot rules, 1% resize band, T+1/locks/settlement and same-open execution assumption; not upstream 5000 currency resize band or liquidity-participation cap.',
            'Reserve locked holdings before allocation, scaling down free targets only if needed; realized positions/weights can exceed desired30/5% due locks or drift.',
            '2017 stopping and previously studied feature selection are not independent; external intervals observation only.']}
    assert frozen['fees']==protocol['fees']
    atomic_write_json(out/'frozen.json',frozen);rows=[];results=[]
    history=load_signal_history(root,model_sources,config)
    np.savez_compressed(out/'full_signal_history.npz',**{k:history[k] for k in ['dates','dispersion','percentile','scale']})
    history_checks=[]
    for split in ('fit','internal_validation','train','validation','test'):
        prepared={h:PreparedModelPortfolio(p,h,split) for h,p in model_sources.items()}
        rt=prepared[5].runtime
        for h,p in prepared.items():
            np.testing.assert_array_equal(rt.trade_dates,p.runtime.trade_dates)
            np.testing.assert_array_equal(rt.stock_codes,p.runtime.stock_codes)
        positions=np.searchsorted(history['dates'],rt.trade_dates);matched=positions<len(history['dates'])
        matched[matched]&=history['dates'][positions[matched]]==rt.trade_dates[matched]
        assert matched[rt.decision_start:rt.decision_stop].all()
        liquidity=np.zeros((rt.n_dates,rt.n_stocks),bool);vol=np.full(liquidity.shape,np.nan)
        liquidity[matched]=history['liquidity'][positions[matched]];vol[matched]=history['volatility'][positions[matched]]
        preds={h:np.where(liquidity,p.prediction,np.nan) for h,p in prepared.items()}
        dispersion=np.full(rt.n_dates,np.nan);percentile=dispersion.copy();scale=np.full(rt.n_dates,.75)
        for target,key in [(dispersion,'dispersion'),(percentile,'percentile'),(scale,'scale')]:target[matched]=history[key][positions[matched]]
        first=int(positions[rt.decision_start]);count=int(np.isfinite(history['dispersion'][max(0,first-config.trailing+1):first+1]).sum())
        if split in ('internal_validation','validation','test'):assert count==config.trailing,(split,count)
        history_checks.append({'split':split,'first_date':str(rt.trade_dates[rt.decision_start]),'first_percentile_valid_count':count,'full_history_slice_exact':True})
        np.savez_compressed(out/f'{split}_signal_context.npz',dates=rt.trade_dates,dispersion=dispersion,percentile=percentile,scale=scale,liquidity_count=liquidity.sum(axis=1))
        accounts=run_accounts({h:p.episode for h,p in prepared.items()},preds,vol,scale,protocol['initial_cash'],config)
        for name,r in accounts.items():
            trace=r['trace'];metrics=trace.metrics.as_dict()
            np.testing.assert_allclose(trace.portfolio_returns,trace.nav[1:]/trace.nav[:-1]-1,rtol=1e-10,atol=1e-12)
            metrics.update(mean_exposure=float(trace.exposure.mean()),mean_cash_fraction=float((1-trace.exposure).mean()),
                mean_gross_turnover=trace.sum_gross_turnover_ratio/len(trace.portfolio_returns),total_fees=trace.total_fees,
                max_positions=max(d['positions'] for d in r['daily']))
            prefix=out/f'{name}_{split}'
            np.savez_compressed(prefix.with_suffix('.npz'),nav=trace.nav,returns=trace.portfolio_returns,
                exposure=trace.exposure,cash=trace.cash,decision_dates=trace.decision_dates,next_dates=trace.next_decision_dates)
            target_days=np.array(sorted(r['targets']),int)
            np.savez_compressed(out/f'{name}_{split}_targets.npz',day=target_days,weights=np.stack([r['targets'][d] for d in target_days]))
            write_csv(out/f'{name}_{split}_fills.csv',r['fills']);write_csv(out/f'{name}_{split}_daily.csv',r['daily'])
            result=dict(strategy=name,split=split,interval=protocol[split],metrics=metrics,audit=r['audit'],fills=len(r['fills']),
                trace_sha256=file_sha256(prefix.with_suffix('.npz')),target_sha256=file_sha256(out/f'{name}_{split}_targets.npz'))
            atomic_write_json(prefix.with_suffix('.json'),result);results.append(result)
            rows.append(dict(strategy=name,split=split,**metrics))
            print(dict(split=split,strategy=name,cagr=metrics['annualized_return'],calmar=metrics['calmar'],audit=r['audit']),flush=True)
        del prepared,accounts,rt,preds,vol,liquidity;gc.collect()
    for p,sha in sources.items():assert file_sha256(p)==sha,p
    assert file_sha256('configs/config.json')==frozen['config_sha256']
    write_csv(out/'portfolio_results.csv',rows)
    atomic_write_json(out/'history_checks.json',{'state':'pass','splits':history_checks})
    atomic_write_json(out/'complete.json',{'state':'complete','results':results,'seconds':time.monotonic()-started,'frozen_sha256':file_sha256(out/'frozen.json')})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    evaluate(ap.parse_args().root)
