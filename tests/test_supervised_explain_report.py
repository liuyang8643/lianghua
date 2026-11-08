import numpy as np
from test_backtest_lightweight import write_canonical_runtime
from offline_data import load_runtime_slice
from ai.supervised.portfolio import PreparedModelPortfolio
from testback.backtest import trace_to_report_result,core_report_metrics
from ai.supervised.explainability import ReplayAudit,tree_contributions

def test_observer_and_shared_report_preserve_native_account(tmp_path):
    path=tmp_path/'runtime.npz';write_canonical_runtime(path,stocks=30)
    rt=load_runtime_slice(path,'2020-06-01','2020-06-24')
    scores=np.sin(np.arange(rt.n_dates)[:,None]*.7+np.arange(rt.n_stocks)[None,:])
    prepared=PreparedModelPortfolio.from_scores(rt,scores,initial_cash=1e6,source_hash='fixture',split='audit')
    config={'buy_n':5,'sell_m':10,'hold_thresh':3,'max_replacements':1}
    plain,_,_=prepared.evaluate('rank_buffer',config,details=True)
    calls=[]
    observed,_,_=prepared.evaluate('rank_buffer',config,details=True,on_transition=lambda before,step:calls.append((before,step)))
    np.testing.assert_array_equal(plain.nav,observed.nav)
    np.testing.assert_array_equal(plain.cash,observed.cash)
    assert plain.fills==observed.fills and len(calls)==len(observed.portfolio_returns)
    assert calls[0][0].cash==1e6 and calls[-1][1].step_result.account_state==prepared.final_account
    report=trace_to_report_result(observed,prepared.final_account,canonical_config=config)
    metrics=core_report_metrics(report)
    assert report['nav_dates']==list(observed.next_decision_dates)
    assert report['trade_dates']==list(observed.decision_dates)
    assert report['final_asset']==observed.nav[-1]
    assert metrics['total_trades']==sum(map(len,observed.fills))
    assert np.isclose(metrics['total_fees'],sum(f.fee for day in observed.fills for f in day))
    expected_turnover=[sum(f.price*f.quantity for f in fills)/observed.nav[i] for i,fills in enumerate(observed.fills)]
    assert expected_turnover[0]>0
    np.testing.assert_allclose([d['rebalance_funds_ratio'] for d in report['daily_snapshots']],expected_turnover)

def test_saved_pnl_attribution_reconciles_corporate_rebase(tmp_path):
    path=tmp_path/'runtime.npz';write_canonical_runtime(path,stocks=30)
    with np.load(path) as z:data={k:z[k] for k in z.files}
    at=int(np.searchsorted(data['trade_dates'],np.datetime64('2020-06-08')))
    for field in ['open','high','low','close','preClose']:data[field][at:,0]*=.99
    data['preClose'][at,0]=data['close'][at-1,0]*.99
    np.savez(path,**data);rt=load_runtime_slice(path,'2020-06-01','2020-06-24')
    scores=np.broadcast_to(np.arange(30,0,-1),(rt.n_dates,30)).copy()
    prepared=PreparedModelPortfolio.from_scores(rt,scores,initial_cash=1e6,source_hash='fixture',split='audit')
    config={'buy_n':5,'sell_m':10,'hold_thresh':3,'max_replacements':1}
    audit=ReplayAudit(tmp_path,prepared,config)
    trace,_,_=prepared.evaluate('rank_buffer',config,details=True,on_transition=audit)
    audit.finish()
    assert any(x['corporate_action'] for x in audit.positions)
    assert max(audit.reconciliation_errors)<1e-6
    assert sum(x['net_pnl'] for x in audit.positions)==__import__('pytest').approx(trace.nav[-1]-trace.nav[0])

def test_tree_contributions_include_task_and_bias():
    import lightgbm as lgb
    x=np.array([[i/20,i%2] for i in range(80)],float);y=x[:,0]*.1+x[:,1]*.3
    model=lgb.train({'objective':'regression','verbosity':-1,'num_threads':1,'min_data_in_leaf':2},lgb.Dataset(x,label=y,categorical_feature=[1]),num_boost_round=5)
    c,p=tree_contributions(model,x[:7]);assert c.shape==(7,3)
    np.testing.assert_allclose(c.sum(axis=1),p,atol=1e-12)

def test_static_observer_ledger_preserves_native_execution(tmp_path):
    from env.backtest import EpisodeSession,PreparedEpisode,run_day_config_episode
    from env.action_schema import ActionSchema
    from testback.economic_positions import EconomicPositionLedger
    path=tmp_path/'runtime.npz';write_canonical_runtime(path,stocks=30)
    rt=load_runtime_slice(path,'2020-06-01','2020-06-24')
    scores=np.sin(np.arange(rt.n_dates)[:,None]*.7+np.arange(rt.n_stocks)[None,:])
    prepared=PreparedModelPortfolio.from_scores(rt,scores,initial_cash=1e6,source_hash='fixture',split='audit')
    schema=ActionSchema(factor_names=prepared.factors.factor_names,filter_names=prepared.factors.filter_names,fixed_buy_n=5,fixed_sell_m=10,schema_version='economic-ledger-fixture')
    prepared.episode=PreparedEpisode.build(rt,prepared.factors,prefilter_n=30,encode_observations=False)
    day=schema.decode(np.ones(schema.action_dim));first=EpisodeSession(prepared.episode,action_schema=schema,initial_cash=1e6)
    plain=run_day_config_episode(first,lambda _:day)
    second=EpisodeSession(prepared.episode,action_schema=schema,initial_cash=1e6);ledger=EconomicPositionLedger()
    trace=run_day_config_episode(second,lambda _:day,on_transition=ledger)
    np.testing.assert_array_equal(trace.nav,plain.nav);np.testing.assert_array_equal(trace.cash,plain.cash);assert trace.fills==plain.fills
    assert ledger.closed
    assert ledger.summary(second.current_account)['account_pnl']==__import__('pytest').approx(trace.nav[-1]-trace.nav[0])
    from ai.supervised.reporting import write_detailed_observation
    import json
    (tmp_path/'protocol.json').write_text(json.dumps({'initial_cash':1e6,'train':['2020-06-01','2020-06-24'],'observation_splits':['train']}),encoding='utf8')
    (tmp_path/'active_data.json').write_text(json.dumps({'manifest_path':str(tmp_path/'manifest.json')}),encoding='utf8')
    path=write_detailed_observation(tmp_path,'ml','train',rt,trace,second.current_account,{'buy_n':5,'weights':{'MultiHorizonScore':1.},'horizon_weights_pct':{'3':20,'15':50,'30':30}},ledger)
    text=path.read_text(encoding='utf8');data=json.loads(text.split('<script id="report-data" type="application/json">',1)[1].split('</script>',1)[0])
    assert len(data['tables']['cleared']['rows'])==len(ledger.closed)
    assert data['summary']['position_lot_analytics_available'] is True and data['summary']['kline_position_state_available'] is False
    assert sum(data['charts']['winloss'].values())==len(ledger.closed)
    assert data['summary']['final_asset']==trace.nav[-1]
    assert '经济口径' in data['tables']['cleared']['accounting_note']
    for item,row in zip(ledger.closed,data['tables']['cleared']['rows']):
        assert f'data-kline-event="{item["first_event_id"]}"' in row[0]['html']
        trade=trace_to_report_result(trace,second.current_account,canonical_config={})['trade_log'][item['first_event_id']]
        assert trade['code']==item['code'] and trade['action']=='buy'
