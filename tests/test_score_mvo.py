import numpy as np
import pytest
from env.score_mvo import ScoreMVOConfig,ScoreMVOPortfolio,scale_scores


def test_score_scaling_matches_risk_and_positive_affine_score_transforms():
    score=np.array([.2,.4,.7,.8]);risk=np.array([.01,.02,.03,.04])
    a=scale_scores(score,risk)
    assert a.mean()==pytest.approx(0,abs=1e-15)
    assert a.std()==pytest.approx(np.sqrt(risk.mean()))
    np.testing.assert_allclose(a,scale_scores(4*score+30,risk))
    np.testing.assert_array_equal(scale_scores(np.ones(4),risk),np.zeros(4))
    assert np.isfinite(scale_scores([np.nan,1.,2.,3.],risk)).all()
    with pytest.raises(ValueError):scale_scores([np.nan]*4,risk)


def test_mvo_cold_start_cap_and_no_future_risk_rows():
    from test_rl_day_planner import _market
    from env.contracts import AccountState
    market=_market();n=len(market.stock_codes);score=np.arange(n,dtype=float)
    cfg=ScoreMVOConfig(pool_size=n,max_weight=1.,lookback=20)
    history=np.random.default_rng(8).normal(0,.01,(50,n))
    p=ScoreMVOPortfolio(market.stock_codes,config=cfg)
    w=p.targets(market,AccountState(cash=1e6,nav=1e6),30,score,history[:30])
    assert sum(w.values())==pytest.approx(1,abs=1e-8)
    assert p.diagnostics[-1]['effective_turnover']==1
    np.testing.assert_array_equal(p.last_inputs['history'],history[10:30,p.last_inputs['indices']])
    history[30:]=1000
    q=ScoreMVOPortfolio(market.stock_codes,config=cfg)
    w2=q.targets(market,AccountState(cash=1e6,nav=1e6),30,score,history[:30])
    assert w==w2


def test_mvo_fully_locked_overweight_is_preserved_without_adding():
    from test_rl_day_planner import _market
    from env.contracts import AccountState
    market=_market();n=len(market.stock_codes)
    account=AccountState(cash=0,nav=10000,positions={'600000.SH':1000},sellable_positions={})
    p=ScoreMVOPortfolio(market.stock_codes)
    w=p.targets(market,account,1,np.arange(n,dtype=float),np.random.default_rng(9).normal(0,.01,(120,n)))
    assert w['600000.SH']==pytest.approx(1,abs=1e-8)
    assert p.last_inputs['weights'][0]>=p.last_inputs['lower'][0]-1e-8


def test_mvo_empty_buyable_pool_keeps_cash_and_records_empty_inputs():
    from dataclasses import replace
    from test_rl_day_planner import _market,_config
    from env.contracts import AccountState,PortfolioExecutionSpec
    from env.planner import DayPlanner
    market=_market();n=len(market.stock_codes)
    market=replace(market,open_prices=np.zeros(n))
    account=AccountState(cash=1e6,nav=1e6)
    p=ScoreMVOPortfolio(market.stock_codes)
    target=p.targets(market,account,0,np.ones(n),np.empty((0,n)))
    assert target=={} and p.last_inputs['indices'].size==0
    assert p.last_inputs['history'].shape==(0,0)
    assert p.diagnostics[-1]['target_exposure']==0 and p.diagnostics[-1]['solver']=='no_solve'
    plan=DayPlanner().plan(market,account,_config(),target_weights=target,portfolio_spec=PortfolioExecutionSpec(n,1.,allow_cash=True))
    assert not plan.buy_orders and not plan.sell_orders


def test_mvo_unobserved_held_signal_preserves_existing_account():
    from test_rl_day_planner import _market
    from env.contracts import AccountState
    market=_market();n=len(market.stock_codes)
    account=AccountState(cash=5000,nav=10000,positions={'600000.SH':500},sellable_positions={'600000.SH':500})
    p=ScoreMVOPortfolio(market.stock_codes)
    target=p.targets(market,account,0,np.full(n,np.nan),np.empty((0,n)))
    assert target=={'600000.SH':.5}
    assert p.diagnostics[-1]['status']=='hold_no_observable_signal'


def test_mvo_real_native_account_execution(tmp_path):
    from test_backtest_lightweight import write_canonical_runtime
    from offline_data import load_runtime_slice
    from factor import precompute_factors
    from env.backtest import PreparedEpisode,EpisodeSession,run_target_weight_episode
    from env.action_schema import ActionSchema
    from env.contracts import PortfolioExecutionSpec
    from env.forward_returns import open_forward_returns
    path=tmp_path/'runtime.npz';write_canonical_runtime(path,stocks=30)
    rt=load_runtime_slice(path,'2020-06-01','2020-06-24',preload_rows=100001)
    factors=precompute_factors(rt);episode=PreparedEpisode.build(rt,factors,encode_observations=False)
    schema=ActionSchema(factor_names=factors.factor_names,filter_names=factors.filter_names,fixed_buy_n=30,fixed_sell_m=30,fixed_turnover_rate=1.)
    session=EpisodeSession(episode,action_schema=schema,portfolio_spec=PortfolioExecutionSpec(30,1.,allow_cash=True))
    strategy=ScoreMVOPortfolio(rt.stock_codes,config=ScoreMVOConfig(pool_size=20,max_weight=.1,lookback=20))
    scores=np.cos(np.arange(rt.n_dates)[:,None]*.3+np.arange(rt.n_stocks)[None,:]);returns=open_forward_returns(rt.data,1)
    def provider(ctx):
        i=ctx.current_index
        return strategy.targets(ctx.current_market,ctx.current_account,i,scores[i],returns[:i])
    trace=run_target_weight_episode(session,schema.decode(np.ones(schema.action_dim)),provider,record_details=False)
    assert trace.total_fees>0 and trace.full_investment_contract_satisfied
    assert all(d['target_l1_turnover']<=d['effective_turnover']+1e-8 for d in strategy.diagnostics)
    assert min(trace.cash)>=0 and len(strategy.diagnostics)==len(trace.portfolio_returns)
