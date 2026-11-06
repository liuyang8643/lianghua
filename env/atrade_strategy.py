"""ATrade baseline_v2 allocation rules adapted to native WBR execution.

Reference: Hzlvv/ATrade 03bda3a, atrade/ml/{portfolio,runner,timing}.py.
Local open-time predictors, exchange rules and fees deliberately remain local.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from env.qlib_strategy import _PortfolioState
from env.quantity import minimum_buy_quantity
from env.fees import DEFAULT_FEE_SCHEDULE


@dataclass(frozen=True)
class ATradeConfig:
    topk:int=30
    buffer:int=60
    cap:float=.05
    rebalance:int=5
    volatility_floor:float=.005
    trailing:int=250
    minimum_history:int=60
    exposure_low:float=.5
    exposure_high:float=1.
    liquidity_top_n:int=500

    def __post_init__(self):
        if any(type(x) is not int for x in (self.topk,self.buffer,self.rebalance,self.trailing,self.minimum_history,self.liquidity_top_n)):
            raise TypeError('count parameters must be integers')
        if not(1<=self.topk<=self.buffer and self.rebalance>0 and 1<=self.minimum_history<=self.trailing and self.liquidity_top_n>0):
            raise ValueError('invalid counts')
        if not(0<self.cap<=1 and self.volatility_floor>0 and 0<=self.exposure_low<=self.exposure_high<=1):
            raise ValueError('invalid weights')


def completed_volatility(close,preclose,member):
    """20 completed daily economic close returns; T close never enters T signal."""
    valid=np.isfinite(close)&np.isfinite(preclose)&(close>0)&(preclose>0)&member
    returns=np.divide(close,preclose,out=np.full_like(close,np.nan,dtype=float),where=valid)-1
    return pd.DataFrame(returns).rolling(20,min_periods=15).std().shift(1).to_numpy()


def completed_liquidity_pool(amount,member,top_n=500):
    """Upstream average-rank top500 rule shifted to local T-open information."""
    valid=np.isfinite(amount)&(amount>=0)&member
    average=pd.DataFrame(np.where(valid,amount,np.nan)).rolling(20,min_periods=20).mean().shift(1)
    return (average.rank(axis=1,ascending=False).to_numpy()<=top_n)&member


def dispersion_exposure(predictions,config=ATradeConfig()):
    """Equal-horizon mean cross-sectional SD, then causal rolling percentile."""
    if len(predictions)!=3:raise ValueError('requires the three 5/20/60 panels')
    if len({np.shape(x) for x in predictions})!=1:raise ValueError('prediction axes differ')
    dispersion=pd.concat([pd.DataFrame(p).std(axis=1) for p in predictions],axis=1).mean(axis=1)
    percentile=dispersion.rolling(config.trailing,min_periods=config.minimum_history).rank(pct=True)
    scale=config.exposure_low+(config.exposure_high-config.exposure_low)*percentile
    scale=scale.clip(config.exposure_low,config.exposure_high).fillna((config.exposure_low+config.exposure_high)/2)
    return dispersion.to_numpy(),percentile.to_numpy(),scale.to_numpy()


def inverse_volatility_target(scores,volatility,held,buyable,minimum_cost,equity,config=ATradeConfig()):
    """Keep held top-buffer names, refill to topk, cap inverse-vol weights.

    Infeasible new entries are removed before recomputing weights; caps are
    never renormalized upward. minimum_cost uses native board lots and fees.
    """
    scores=np.asarray(scores);volatility=np.asarray(volatility);held=np.asarray(held,bool)
    valid=np.isfinite(scores)&np.isfinite(volatility)&(held|buyable)
    order=np.argsort(-np.where(valid,scores,-np.inf),kind='stable');order=order[valid[order]]
    retained=order[:config.buffer][held[order[:config.buffer]]][:config.topk]
    rejected=np.zeros(len(scores),bool)
    while True:
        remaining=order[~np.isin(order,retained)&~rejected[order]]
        selected=np.r_[retained,remaining[:config.topk-len(retained)]]
        weights=np.zeros(len(scores))
        if not len(selected):return weights
        inverse=1/np.maximum(volatility[selected],config.volatility_floor)
        weights[selected]=np.minimum(inverse/inverse.sum(),config.cap)
        new=selected[~held[selected]]
        bad=new[(weights[new]*equity<minimum_cost[new])|~np.isfinite(minimum_cost[new])]
        if not len(bad):return weights
        rejected[bad]=True


def executable_targets(desired,current,sellable,buyable):
    """Reserve fully locked holdings and restrict increases to legal buys.

    This is an execution projection, not a reoptimization: scale free weights
    down only if locked NAV needs space. Never fill intentionally empty cash.
    """
    desired=np.asarray(desired,float);current=np.asarray(current,float)
    if desired.shape!=current.shape or not np.isfinite(desired).all() or np.any(desired<0) or desired.sum()>1+1e-8:
        raise ValueError('invalid allocation target')
    minimum=np.where(sellable,0.,current)
    result=np.maximum(desired,minimum)
    result=np.where(buyable,result,np.minimum(result,current))
    if result.sum()>1:
        slack=np.maximum(result-minimum,0.)
        result=minimum+slack*(max(0.,1-minimum.sum())/max(slack.sum(),1e-30))
    return result


class ATradeBookPortfolio(_PortfolioState):
    def __init__(self,codes,config=ATradeConfig()):
        super().__init__(codes);self.config=config
        self.minimums=np.asarray([minimum_buy_quantity(c) for c in codes])

    def targets(self,market,account,day,scores,volatility,*,rebalance,desired=None):
        held,sellable,_,current,_,buyable,equity=self.context(market,account,day,scores)
        if not rebalance:return {self.codes[i]:float(current[i]) for i in np.flatnonzero(current>0)}
        if desired is None:
            notionals=self.minimums*market.open_prices
            fees=DEFAULT_FEE_SCHEDULE
            costs=notionals+np.maximum(notionals*fees.commission_rate,fees.minimum_commission)+notionals*(fees.transfer_fee_rate+fees.slippage_rate)
            desired=inverse_volatility_target(scores,volatility,held,buyable,costs,equity,self.config)
        self.last_desired=np.array(desired,copy=True)
        actual=executable_targets(desired,current,sellable,buyable)
        self.diagnostics.append({'day':int(day),'desired_count':int(np.count_nonzero(desired)),
            'desired_exposure':float(desired.sum()),'projected_count':int(np.count_nonzero(actual)),
            'projected_exposure':float(actual.sum()),'locked_weight':float(current[~sellable].sum())})
        return {self.codes[i]:float(actual[i]) for i in np.flatnonzero(actual>0)}
