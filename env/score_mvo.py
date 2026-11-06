"""Risk-scaled prediction scores -> ordinary MVO using existing WBR QP solvers.

Like Qlib PortfolioOptimizer(scale_return=True), the score is an alpha utility,
not an estimated percentage return. No realized label is used for calibration.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from env.qlib_strategy import _PortfolioState,shrunk_covariance,solve_mean_variance
from env.factorized_mvo import risk_components,solve_factor


@dataclass(frozen=True)
class ScoreMVOConfig:
    pool_size:int=50
    risk_aversion:float=10.
    turnover_limit:float=.1
    ridge:float=.001
    lookback:int=120
    shrinkage:float=.5
    max_weight:float=.05

    def __post_init__(self):
        if type(self.pool_size) is not int or self.pool_size<1 or type(self.lookback) is not int or self.lookback<2:
            raise ValueError('invalid MVO pool or risk window')
        if not all(np.isfinite(x) and x>=0 for x in (self.risk_aversion,self.turnover_limit,self.ridge)):
            raise ValueError('invalid MVO objective parameters')
        if not 0<=self.shrinkage<=1 or not 0<self.max_weight<=1:raise ValueError('invalid MVO bounds')


def scale_scores(scores,risk_diagonal):
    """Centering is objective-invariant under sum(w)=1; finite constant scores give zero alpha."""
    s=np.asarray(scores,float);d=np.asarray(risk_diagonal,float)
    if s.ndim!=1 or d.shape!=s.shape or not np.isfinite(d).all() or np.any(d<0):raise ValueError('invalid signal/risk axes')
    good=np.isfinite(s)
    if not good.any():raise ValueError('no observable candidate signal')
    s=np.where(good,s,np.min(s[good]));sd=float(s.std())
    return (s-s.mean())/sd*np.sqrt(float(d.mean())) if sd>1e-12 else np.zeros_like(s)


class ScoreMVOPortfolio(_PortfolioState):
    def __init__(self,codes,horizon=5,config=ScoreMVOConfig()):
        super().__init__(codes)
        if type(horizon) is not int or horizon<1:raise ValueError('invalid horizon')
        self.horizon=horizon;self.config=config

    def targets(self,market,account,day,scores,observed_returns):
        started=time.monotonic();cfg=self.config
        held,_,_,current,locked,buyable,equity=self.context(market,account,day,scores)
        order=np.argsort(-np.where(np.isfinite(scores),scores,-np.inf),kind='stable')
        indices=np.unique(np.r_[np.flatnonzero(held),order[buyable[order]][:cfg.pool_size]])
        history=np.asarray(observed_returns[-cfg.lookback:,indices],float)
        w0=current[indices];lower=locked[indices]
        # Existing drift/locked overweight can remain, but is never increased.
        # Newly established/increased positions cannot exceed the5% target cap.
        upper=np.where(buyable[indices],np.maximum(cfg.max_weight,w0),w0)
        funding=max(0.,1-float(w0.sum()));limit=max(cfg.turnover_limit,funding)
        observable=np.isfinite(scores[indices]).any()
        if not len(indices) or upper.sum()<1-1e-8 or not observable:
            weights=w0.copy();diagnostic={'status':'hold_insufficient_legal_capacity','solver':'no_solve'}
            if not observable and len(indices):diagnostic['status']='hold_no_observable_signal'
            alpha=np.zeros(len(indices));variance=np.zeros(len(indices));counts=np.isfinite(history).sum(axis=0)
        else:
            if len(history)<2:raise ValueError('insufficient past risk returns')
            factorized=len(indices)>=256
            if factorized:
                factor,diagonal,counts=risk_components(history,shrinkage=cfg.shrinkage,horizon=self.horizon)
                variance=np.sum(factor*factor,axis=0)+diagonal
            else:
                covariance,counts=shrunk_covariance(history,shrinkage=cfg.shrinkage,horizon=self.horizon)
                variance=np.diag(covariance)
            alpha=scale_scores(scores[indices],variance)
            kwargs=dict(risk_aversion=cfg.risk_aversion,turnover_limit=limit,ridge=cfg.ridge,lower=lower,upper=upper)
            if factorized:weights,diagnostic=solve_factor(alpha,factor,diagonal,w0,**kwargs)
            else:weights,diagnostic=solve_mean_variance(alpha,covariance,w0,**kwargs)
            if weights is None:raise RuntimeError('MVO numerical failure without constraint relaxation: '+str(diagnostic))
        if np.any(weights<lower-1e-8) or np.any(weights>upper+1e-8) or np.abs(weights-w0).sum()>limit+1e-8:
            raise AssertionError('MVO target violates portfolio constraints')
        self.last_inputs=dict(indices=indices.copy(),alpha=alpha.copy(),risk_diagonal=variance.copy(),
            current=w0.copy(),lower=lower.copy(),upper=upper.copy(),weights=weights.copy(),
            history=history.copy(),turnover_limit=np.array(limit),equity=np.array(equity))
        diagnostic.update(day=int(day),candidates=len(indices),holdings=int(held.sum()),risk_rows=len(history),
            minimum_risk_observations=int(counts.min()) if len(counts) else 0,target_count=int(np.count_nonzero(weights)),target_exposure=float(weights.sum()),
            target_l1_turnover=float(np.abs(weights-w0).sum()),configured_turnover=cfg.turnover_limit,
            cash_funding_floor=funding,effective_turnover=limit,drift_cap_exceptions=int(np.sum(w0>cfg.max_weight+1e-8)),
            raw_score_std=float(np.nanstd(scores[indices])) if observable else 0.,alpha_std=float(alpha.std()) if len(alpha) else 0.,seconds=time.monotonic()-started)
        self.diagnostics.append(diagnostic)
        return {self.codes[i]:float(w) for i,w in zip(indices,weights) if w>0}
