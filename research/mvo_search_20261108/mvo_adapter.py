"""Frozen rank utility to ordinary correlated MVO; local account stays authoritative."""
import numpy as np
from scipy.stats import rankdata
from env.qlib_strategy import _PortfolioState
from env.factorized_mvo import risk_components,solve_factor_interior
from env.action_schema import ActionSchema
from env.backtest import EpisodeSession,run_target_weight_episode
from env.contracts import PortfolioExecutionSpec
from env.fees import DEFAULT_FEE_SCHEDULE

class Strategy(_PortfolioState):
    def __init__(self,codes,p):super().__init__(codes);self.p=p
    def targets(self,market,account,day,scores,past):
        held,sellable,ages,current,locked,buyable,equity=self.context(market,account,day,scores)
        good=np.isfinite(scores)&(market.listing_age>=0)&~market.delisted_mask&~market.st_mask&(market.open_prices>=2)
        utility=np.zeros(len(scores));percent=(rankdata(scores[good])-.5)/good.sum()
        utility[good]=(percent-percent.mean())/max(percent.std(),1e-12)
        order=np.argsort(-np.where(np.isfinite(scores),scores,-np.inf),kind='stable')
        ix=np.unique(np.r_[np.flatnonzero(held),order[buyable[order]][:self.p['pool_size']]])
        f,d,counts=risk_components(past[-120:,ix],shrinkage=.5,horizon=1)
        scale=np.sqrt(np.median(np.sum(f*f,axis=0)+d));r=utility[ix]*scale
        w0=current[ix];lo=locked[ix].copy();cap=self.p['max_weight']
        hi=np.where(buyable[ix],np.maximum(cap,w0),w0)
        funding=max(0.,1-w0.sum());limit=max(self.p['turnover_limit'],funding)
        if not len(ix) or hi.sum()<1-1e-10:raise ValueError('insufficient legal capacity under weight cap')
        w,diag=solve_factor_interior(r,f,d,w0,risk_aversion=self.p['risk_aversion'],turnover_limit=limit,ridge=1e-6,lower=lo,upper=hi)
        if w is None:raise RuntimeError('MVO failed without relaxation '+str(diag))
        assert abs(w.sum()-1)<1e-8 and np.max(w-hi)<1e-8 and np.max(lo-w)<1e-8
        self.diagnostics.append({'day':int(day),'candidates':len(ix),'configured_turnover':self.p['turnover_limit'],'effective_turnover':float(limit),'funding_floor':float(funding),'locked_over_cap':int(sum(lo>cap+1e-10)),'drift_over_cap':int(sum(w0>cap+1e-10)),'min_risk_observations':int(counts.min()),**diag})
        return {self.codes[j]:float(v) for j,v in zip(ix,w) if v>0}

def evaluate(prepared,p,observed,*,details=False,on_transition=None):
    strategy=Strategy(prepared.runtime.stock_codes,p);limit=prepared.runtime.n_stocks
    schema=ActionSchema(factor_names=prepared.factors.factor_names,filter_names=prepared.factors.filter_names,fixed_buy_n=limit,fixed_sell_m=limit,fixed_turnover_rate=1.,fixed_rebalance_band_pct=.01,schema_version='supervised-portfolio-day-v1')
    session=EpisodeSession(prepared.episode,action_schema=schema,initial_cash=prepared.initial_cash,fees=DEFAULT_FEE_SCHEDULE,portfolio_spec=PortfolioExecutionSpec(limit,1.,allow_cash=False))
    def targets(context):
        i=context.current_index
        # Return row t uses open[t+1]. Exclude row i-1, which ends at today's open.
        return strategy.targets(context.current_market,context.current_account,i,prepared.prediction[i],observed[:max(0,i-1)])
    trace=run_target_weight_episode(session,schema.decode(np.ones(schema.action_dim)),targets,record_details=details,on_transition=on_transition)
    m=trace.metrics.as_dict();m.update(total_fees=float(trace.total_fees),mean_gross_turnover=float(trace.sum_gross_turnover_ratio/len(trace.portfolio_returns)),mean_exposure=float(trace.exposure.mean()),funding_floor_days=sum(d['effective_turnover']>d['configured_turnover']+1e-8 for d in strategy.diagnostics),solver_days=len(strategy.diagnostics))
    return trace,m,strategy.diagnostics,session.current_account
