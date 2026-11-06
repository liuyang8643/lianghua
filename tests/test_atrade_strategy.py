import numpy as np
import pytest
from env.atrade_strategy import (ATradeConfig,inverse_volatility_target,executable_targets,
    completed_volatility,completed_liquidity_pool,dispersion_exposure,ATradeBookPortfolio)


def test_buffer_retains_actual_held_stock_and_cap_leaves_cash():
    cfg=ATradeConfig(topk=2,buffer=4,cap=.4)
    score=np.arange(6.,0.,-1);held=np.array([0,0,1,0,0,1],bool)
    w=inverse_volatility_target(score,np.ones(6)*.02,held,np.ones(6,bool),np.ones(6),100,cfg)
    np.testing.assert_array_equal(np.flatnonzero(w),[0,2])
    assert w.sum()==pytest.approx(.8)


def test_inverse_volatility_and_unaffordable_head_replaced():
    cfg=ATradeConfig(topk=2,buffer=3,cap=1.)
    w=inverse_volatility_target(np.array([3.,2.,1.]),np.array([.02,.01,.02]),np.zeros(3,bool),
        np.ones(3,bool),np.array([1000.,1.,1.]),100,cfg)
    np.testing.assert_allclose(w,[0,2/3,1/3])


def test_lock_projection_preserves_locked_nav_and_does_not_fill_cash():
    w=executable_targets([0,.7,0],[.4,0,.1],[False,True,True],[False,True,False])
    np.testing.assert_allclose(w,[.4,.6,0])
    np.testing.assert_allclose(executable_targets([.2,.1],[.2,0],[True,True],[True,True]),[.2,.1])
    np.testing.assert_allclose(executable_targets([.1,.8],[.2,0],[False,True],[False,False]),[.2,0])


def test_completed_market_inputs_ignore_current_and_future_bars():
    rng=np.random.default_rng(4);close=10+rng.random((50,4));pre=np.full_like(close,10.)
    amount=rng.uniform(10,20,(50,4));member=np.ones_like(close,bool)
    v=completed_volatility(close,pre,member);pool=completed_liquidity_pool(amount,member,2)
    close[30:]*=100;pre[30:]*=.1;amount[30:]*=1000
    np.testing.assert_allclose(completed_volatility(close,pre,member)[:31],v[:31],equal_nan=True)
    np.testing.assert_array_equal(completed_liquidity_pool(amount,member,2)[:31],pool[:31])


def test_dispersion_uses_three_scores_causally_and_midpoint_warmup():
    x=np.column_stack([np.arange(80.),np.zeros(80)])
    _,p,e=dispersion_exposure([x,x.copy(),x.copy()])
    np.testing.assert_allclose(e[:59],.75);assert p[59]==1 and e[59]==1
    y=x.copy();y[70:]*=1000
    np.testing.assert_allclose(dispersion_exposure([y,y,y])[2][:70],e[:70])


def test_chained_liquidity_and_percentile_need_full_pre_split_history():
    rng=np.random.default_rng(2026);amount=rng.uniform(10,100,(800,12));member=np.ones_like(amount,bool)
    predictions=[rng.normal(size=amount.shape) for _ in range(3)]
    def context(start,stop):
        pool=completed_liquidity_pool(amount[start:stop],member[start:stop],5)
        return dispersion_exposure([np.where(pool,p[start:stop],np.nan) for p in predictions])
    full=context(0,800);short=context(600-252,800);enough=context(600-270,800)
    assert np.isfinite(short[0][252-249:253]).sum()==233
    assert np.isfinite(full[0][600-249:601]).sum()==250
    np.testing.assert_allclose(enough[2][270:],full[2][600:])
    # Computing the full history before split selection is causal as well.
    prefix=context(0,650)
    np.testing.assert_allclose(prefix[2],full[2][:650])


def test_nonrebalance_targets_keep_actual_weights_in_native_planner():
    from test_rl_day_planner import _market,_config
    from env.contracts import AccountState,PortfolioExecutionSpec
    from env.planner import DayPlanner
    market=_market();account=AccountState(cash=2000,nav=10000,positions={'600000.SH':800},sellable_positions={'600000.SH':800})
    strategy=ATradeBookPortfolio(market.stock_codes,ATradeConfig(topk=1,buffer=2))
    targets=strategy.targets(market,account,1,np.ones(len(market.stock_codes)),np.ones(len(market.stock_codes))*.02,rebalance=False)
    plan=DayPlanner().plan(market,account,_config(),target_weights=targets,portfolio_spec=PortfolioExecutionSpec(1,1.,allow_cash=True))
    assert not plan.buy_orders and not plan.sell_orders


def test_five_native_accounts_merge_targets_and_hold_between_rebalances(tmp_path):
    from test_backtest_lightweight import write_canonical_runtime
    from offline_data import load_runtime_slice
    from factor import precompute_factors
    from env.backtest import PreparedEpisode
    from ai.supervised.atrade_study import run_accounts
    path=tmp_path/'runtime.npz';write_canonical_runtime(path,stocks=30)
    rt=load_runtime_slice(path,'2020-06-01','2020-06-24',preload_rows=100001)
    episode=PreparedEpisode.build(rt,precompute_factors(rt),encode_observations=False)
    pred={h:np.sin(np.arange(rt.n_dates)[:,None]*.3+np.arange(rt.n_stocks)[None,:]+h) for h in [5,20,60]}
    scale=np.full(rt.n_dates,.75)
    results=run_accounts({h:episode for h in pred},pred,np.ones_like(pred[5])*.02,scale,1e6,
        ATradeConfig(topk=5,buffer=10,cap=.3))
    assert set(results)=={'book5','book20','book60','anchor','timed'}
    for r in results.values():
        assert not any(r['audit'].values()) and len(r['fills'])>0
        assert r['trace'].full_investment_contract_satisfied
    for day,w in results['anchor']['targets'].items():
        np.testing.assert_allclose(w,sum(results[f'book{h}']['targets'][day] for h in [5,20,60])/3)
        np.testing.assert_allclose(results['timed']['targets'][day],w*.75)
    # Allocations with deliberate cash must not be silently swept to 100%.
    assert results['timed']['trace'].exposure[0]<.76
